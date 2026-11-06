"""Sanity checks: does the cleaned data work, and is it usable for the model?"""
import numpy as np
import pandas as pd


def basic_report(census: pd.DataFrame, ruca: pd.DataFrame, tracts_2020: set) -> None:
    """tracts_2020 = set(xwalk['GEOID_TRACT_20']) (drop state '72' if excluding Puerto Rico)."""
    tracts_2020 = {t for t in tracts_2020 if not t.startswith("72")}
    print("== Keys ==")
    print("census duplicate (tract, year) rows:", census.duplicated(["tract_id", "acs_year"]).sum())
    print("ruca duplicate tracts:", ruca["tract_id"].duplicated().sum())
    print("tract_id lengths:", census["tract_id"].str.len().value_counts().to_dict(),
          ruca["tract_id"].str.len().value_counts().to_dict())

    print("\n== Coverage vs. 2020 tract list ==")
    for y, g in census.groupby("acs_year"):
        print(f"ACS {y}: {len(set(g.tract_id) & tracts_2020) / len(tracts_2020):.3f} of 2020 tracts")
    print(f"RUCA: {len(set(ruca.tract_id) & tracts_2020) / len(tracts_2020):.3f} of 2020 tracts")

    print("\n== National population (50 states + DC should be ~300-345M) ==")
    print((census.groupby("acs_year")["pop_total"].sum() / 1e6).round(1).to_string())
    if "pop_conserved_ratio" in census:
        print("\nCrosswalk population conservation (want 0.99-1.01):")
        print(census.groupby("acs_year")["pop_conserved_ratio"].first().round(4).to_string())

    print("\n== Value ranges ==")
    for c in ["pct_nh_white", "pct_nonwhite", "pct_hispanic", "poverty_rate",
              "pct_bachelors_plus", "unemployment_rate"]:
        if c not in census:
            continue
        bad = ((census[c] < -1e-9) | (census[c] > 1 + 1e-9)).sum()
        print(f"{c}: {bad} values outside [0,1]")
    print("non-positive median income:", (census["median_hh_income"] <= 0).sum())
    print("race shares sum > 1.01:",
          (census[["pct_nh_white", "pct_nh_black", "pct_nh_aian", "pct_nh_asian", "pct_hispanic"]]
           .sum(axis=1) > 1.01).sum())

    print("\n== Missingness by column (share NaN, latest year) ==")
    last = census[census.acs_year == census.acs_year.max()]
    last = last.drop(columns=["tract_id", "pop_conserved_ratio"], errors="ignore")
    print(last.isna().mean().round(3).sort_values(ascending=False).head(10).to_string())
    print("\nRUCA classes:", ruca["ruca_class"].value_counts(dropna=False).to_dict())


def dc_join_report(dc: pd.DataFrame, census: pd.DataFrame, ruca: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """dc needs columns: tract_id (2020 GEOID), construction_year. Joins features measured at t - lag
    and reports how many data centers get usable features. Returns the joined table."""
    d = dc.copy()
    d["acs_year"] = (d["construction_year"] - lag).clip(lower=census.acs_year.min(), upper=census.acs_year.max())
    j = d.merge(census, on=["tract_id", "acs_year"], how="left", indicator=True) \
         .merge(ruca, on="tract_id", how="left")
    print(f"Census join rate:  {(j['_merge'] == 'both').mean():.3f}")
    print(f"RUCA join rate:    {j['ruca_class'].notna().mean():.3f}")
    print(f"low-population tracts among data centers: {j['low_pop_flag'].mean():.3f}")
    print("missing income / poverty / pct_nonwhite among data centers:",
          j[["median_hh_income", "poverty_rate", "pct_nonwhite"]].isna().mean().round(3).to_dict())
    return j


def signal_check(cases: pd.DataFrame, controls: pd.DataFrame, features: list) -> pd.Series:
    """Single-feature AUC (0.5 = no signal). Data-center tracts vs control tracts."""
    from sklearn.metrics import roc_auc_score
    x = pd.concat([cases.assign(y=1), controls.assign(y=0)], ignore_index=True)
    res = {}
    for f in features:
        s = x[[f, "y"]].dropna()
        res[f] = roc_auc_score(s["y"], s[f]) if s["y"].nunique() == 2 and len(s) > 20 else np.nan
    return pd.Series(res).sort_values(key=lambda v: (v - 0.5).abs(), ascending=False)