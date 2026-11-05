"""Turn the data center registry into a model-ready table (cases + controls with Census/RUCA features).

Save as src/features/dc_table.py. Run from the repo root:

    # Stage 1: one ACS snapshot (2020), joined to every tract in the registry
    python -m src.features.dc_table data/raw/dc_registry.csv snapshot
s
    # Stage 2: features from the year BEFORE each data center was built (dated sites only)
    python -m src.features.dc_table data/raw/dc_registry.csv prebuild

Stage 1 reads data/interim/snapshot_2020/ (small, built in minutes).
Stage 2 reads data/interim/ (the full 2010-2024 panel).
Outputs are small (a few thousand rows), so these are the files worth committing.
"""
import sys
import numpy as np
import pandas as pd
from .acs_ruca import load_acs_ruca

TERRITORIES = ["PR", "GU", "VI", "MP"]
PIPELINE = ["under_construction", "permitted", "proposed"]
FEATS = ["median_hh_income", "income_pctile_year", "median_home_value", "poverty_rate",
         "pct_nonwhite", "pct_nh_black", "pct_hispanic", "pct_nh_asian", "pct_nh_aian",
         "pct_bachelors_plus", "unemployment_rate", "pop_density_km2", "ruca_primary"]
FIRST, LAST, LAG, SNAPSHOT_YEAR = 2011, 2025, 1, 2020     # config: min/max_construction, feature_lag_years


def load_registry(path):
    """Read the registry. tract_geoid was saved as a number, so restore the leading zero."""
    d = pd.read_csv(path, dtype={"tract_geoid": "string"}, low_memory=False)
    d["tract_id"] = d["tract_geoid"].str.replace(r"\.0$", "", regex=True).str.zfill(11)
    d.loc[d["tract_geoid"].isna(), "tract_id"] = pd.NA
    return d


def sites_to_tracts(d, statuses=None):
    """Clean the site list and collapse to ONE ROW PER TRACT.

    statuses: optional list of operational_status values to keep (e.g. ["operational"]).
              'pnnl_only' means the site has no Compute Atlas status. Default keeps everything but cancelled.
    """
    d = d[d.tract_id.notna() & ~d.state.isin(TERRITORIES)].copy()
    d["status"] = d["operational_status"].fillna("pnnl_only")
    d = d[d.status != "cancelled"]
    if statuses is not None:
        d = d[d.status.isin(statuses)]
    # a year is trustworthy only if NLCD detected the change ('high'/'low') or Epoch AI supplied it;
    # 'built_before_window' and 'no_transition_detected' mean the year is UNKNOWN
    d["dated"] = d.construction_year_confidence.isin(["high", "low"]) | (d.construction_year_source == "epoch_ai")
    d["year"] = d["construction_year"].where(d["dated"])
    t = d.groupby("tract_id").agg(
        n_sites=("site_id", "size"),
        n_operational=("status", lambda s: int((s == "operational").sum())),
        n_pipeline=("status", lambda s: int(s.isin(PIPELINE).sum())),
        n_dated=("dated", "sum"),
        first_year=("year", "min")).reset_index()
    t["label"] = 1
    return t


def make_cases(tracts, mode):
    """The ONLY difference between stage 1 and stage 2: which ACS year each case is joined to."""
    t = tracts.copy()
    if mode == "snapshot":
        t["acs_year"] = SNAPSHOT_YEAR
    elif mode == "prebuild":
        t = t[t.first_year.between(FIRST, LAST)].copy()
        # a tract that also holds an undated (possibly older) site was not "new" in first_year
        n0 = len(t)
        t = t[t.n_dated == t.n_sites].copy()
        print(f"prebuild: dropped {n0 - len(t)} tracts that also contain an undated site")
        t["acs_year"] = (t.first_year - LAG).astype(int)
    else:
        raise ValueError("mode must be 'snapshot' or 'prebuild'")
    t["state_fips"] = t["tract_id"].str[:2]
    t["county_fips"] = t["tract_id"].str[:5]
    return t


def draw_controls(cases, panel, dc_tracts, k=5, seed=42):
    """k controls per case: same state, same ACS year, never a tract with any data center, not tiny."""
    ok = panel[~panel.tract_id.isin(dc_tracts) & ~panel.low_pop_flag & (panel.pop_total > 0)].copy()
    ok["state_fips"] = ok["tract_id"].str[:2]
    pools = {key: g for key, g in ok.groupby(["acs_year", "state_fips"])}
    rng = np.random.default_rng(seed)
    out = []
    for (y, s), grp in cases.groupby(["acs_year", "state_fips"]):
        pool = pools.get((y, s))
        if pool is None:
            continue
        n = min(k * len(grp), len(pool))
        pick = pool.iloc[rng.choice(len(pool), n, replace=False)].copy()
        pick["label"] = 0
        out.append(pick)
    c = pd.concat(out, ignore_index=True)
    c["county_fips"] = c["tract_id"].str[:5]
    return c


def build_table(registry_path, mode, panel, statuses=None, k=5):
    d = load_registry(registry_path)
    tracts = sites_to_tracts(d, statuses)
    print(f"{len(d)} sites -> {len(tracts)} tracts with a data center ({int(tracts.n_sites.sum())} sites kept)")
    cases = make_cases(tracts, mode)
    cases = cases.merge(panel, on=["tract_id", "acs_year"], how="left", indicator=True)
    matched = (cases["_merge"] == "both")
    print(f"mode={mode}: {len(cases)} case tracts, {matched.mean():.3f} found in the ACS table")
    cases = cases[matched].drop(columns="_merge")
    controls = draw_controls(cases, panel, set(tracts.tract_id), k=k)
    df = pd.concat([cases, controls], ignore_index=True)
    print(f"cases {int(df.label.sum())}, controls {int((df.label == 0).sum())}")
    return df


def quick_check(df, feats=FEATS):
    """Do the features separate cases from controls? Cross-validated, folds grouped by county."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    feats = [f for f in feats if f in df]
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42)
    p = cross_val_predict(model, df[feats], df["label"], groups=df["county_fips"],
                          cv=GroupKFold(5), method="predict_proba")[:, 1]
    print(f"\ngradient boosting, 5-fold grouped by county: AUC {roc_auc_score(df.label, p):.3f}, "
          f"avg precision {average_precision_score(df.label, p):.3f} (base rate {df.label.mean():.3f})")
    print("\nsingle-feature AUC (0.5 = no signal):")
    s = {f: roc_auc_score(df.label[df[f].notna()], df[f].dropna()) for f in feats if df[f].nunique() > 1}
    print(pd.Series(s).sort_values(key=lambda v: (v - .5).abs(), ascending=False).round(3).to_string())


def main(registry_path, mode):
    if mode == "snapshot":
        panel = load_acs_ruca("data/interim/snapshot_2020")
    else:
        panel = load_acs_ruca()
    df = build_table(registry_path, mode, panel)
    out = f"data/interim/dc_model_table_{mode}.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out} ({len(df)} rows)")
    quick_check(df)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])