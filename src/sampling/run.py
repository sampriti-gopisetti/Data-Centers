"""Entry point for the 'sampling' stage. Run with: python -m src.sampling.run

Builds matched "no data center" control tracts for Step 2 ("build comparison
dataset") of the proposal, for two purposes:

1. Cumulative: every tract that has ever had a data center (~1,741 tracts),
   matched against similar no-DC tracts at the current (2024) ACS snapshot.
2. Before: the 203 tracts with a trustworthy pre-construction year
   (data/processed/dc_census/dc_sites.csv, rq1_eligible), matched against
   similar no-DC tracts at each tract's own pre-construction ACS year. For the
   44-tract before/after subset (rq4_eligible) the matched controls also get
   their own 2024 snapshot pulled, so a real before/after change comparison
   (change_pop_total, change_poverty_rate, ...) exists for the control side
   too, mirroring dc_tract_before_after.csv's columns for the case side.

Case features come from data/processed/dc_census/dc_tract_acs_long.csv (every
ACS year, every data-center tract -- built by a teammate from the Step 1
registry). Control candidates come from
data/processed/acs_ruca_all_tracts_2010_2024.csv.gz (every US tract, every
year, with a has_data_center flag -- also a teammate's output), filtered down
to tracts that never have a data center (the flag never flips back to False,
so "False in any one year" implies "False in every year"). Matching is
stratified on state + RUCA class, ranked by closest population density,
controls_per_case (config.yaml, sampling.controls_per_case) controls per case.
If a state+RUCA stratum has too few candidates, the state constraint is
dropped (kept: RUCA class nationally) and the row is flagged relaxed_match.
"""
from __future__ import annotations

import pandas as pd

from src.utils import ROOT, load_config

DC_CENSUS_DIR = ROOT / "data" / "processed" / "dc_census"
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLING_DIR = PROCESSED_DIR / "sampling"

CASE_FEATURE_COLS = [
    "pop_total", "households", "median_hh_income", "median_home_value",
    "poverty_rate", "pct_nh_white", "pct_nonwhite", "pct_nh_black",
    "pct_nh_aian", "pct_nh_asian", "pct_hispanic", "pct_bachelors_plus",
    "unemployment_rate", "pop_density_km2", "ruca_primary", "ruca_class",
    "is_rural",
]

# dc_tract_before_after.csv's change_* columns are post - pre on these bases.
CHANGE_BASE_COLS = [
    "pop_total", "pct_nonwhite", "pct_nh_black", "pct_hispanic",
    "pct_nh_asian", "pct_nh_aian", "poverty_rate", "median_hh_income",
    "median_home_value", "pop_density_km2",
]


def _tract_str(series: pd.Series) -> pd.Series:
    """GEOID/tract_id as a zero-padded 11-char string, however it was read in."""
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(11)


def _build_state_lookup() -> pd.Series:
    """tract_id (11-digit str) -> state abbreviation, from EJScreen's full tract universe."""
    ej = pd.read_csv(
        PROCESSED_DIR / "ejscreen_tract_clean_50states_dc.csv",
        dtype={"GEOID": str},
        usecols=["GEOID", "state_abbr"],
    )
    ej["GEOID"] = _tract_str(ej["GEOID"])
    return ej.set_index("GEOID")["state_abbr"]


def _load_control_pool(state_lookup: pd.Series) -> pd.DataFrame:
    pool = pd.read_csv(
        PROCESSED_DIR / "acs_ruca_all_tracts_2010_2024.csv.gz",
        dtype={"tract_id": str},
    )
    pool["tract_id"] = _tract_str(pool["tract_id"])
    ever_dc = pool.groupby("tract_id")["has_data_center"].any()
    never_dc_tracts = set(ever_dc[~ever_dc].index)
    pool = pool[pool["tract_id"].isin(never_dc_tracts)].copy()
    pool["state"] = pool["tract_id"].map(state_lookup)
    return pool


def _match_controls(
    case_tract_id: str,
    case_state: str,
    case_ruca_class: str,
    case_density: float,
    pool_year: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    candidates = pool_year[
        (pool_year["state"] == case_state) & (pool_year["ruca_class"] == case_ruca_class)
    ]
    relaxed = False
    if len(candidates) < k:
        relaxed = True
        candidates = pool_year[pool_year["ruca_class"] == case_ruca_class]

    candidates = candidates.copy()
    candidates["density_diff"] = (candidates["pop_density_km2"] - case_density).abs()
    picked = candidates.sort_values("density_diff", na_position="last").head(k)

    return pd.DataFrame(
        {
            "case_tract_id": case_tract_id,
            "control_rank": range(1, len(picked) + 1),
            "control_tract_id": picked["tract_id"].values,
            "match_year": pool_year["acs_year"].iloc[0],
            "relaxed_match": relaxed,
        }
    )


def _run_matching(cases: pd.DataFrame, pool: pd.DataFrame, k: int) -> pd.DataFrame:
    pool_by_year = {year: df for year, df in pool.groupby("acs_year")}
    matches = []
    for _, case in cases.iterrows():
        pool_year = pool_by_year.get(case["match_year"])
        if pool_year is None:
            print(f"  WARNING: no control pool for year {case['match_year']}, skipping {case['tract_id']}")
            continue
        matches.append(
            _match_controls(
                case["tract_id"], case["state"], case["ruca_class"], case["pop_density_km2"],
                pool_year, k,
            )
        )
    return pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()


def _build_analysis_table(cases: pd.DataFrame, matches: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    case_rows = cases[["tract_id", "match_year"] + CASE_FEATURE_COLS].copy()
    case_rows["is_case"] = True

    control_keys = matches[["control_tract_id", "match_year"]].drop_duplicates()
    control_rows = control_keys.merge(
        pool, left_on=["control_tract_id", "match_year"], right_on=["tract_id", "acs_year"], how="left"
    )
    control_rows = control_rows[["control_tract_id", "match_year"] + CASE_FEATURE_COLS].rename(
        columns={"control_tract_id": "tract_id"}
    )
    control_rows["is_case"] = False

    return pd.concat([case_rows, control_rows], ignore_index=True)


def _cumulative_cases(long_df: pd.DataFrame, state_lookup: pd.Series) -> pd.DataFrame:
    cases = long_df[long_df["acs_year"] == 2024].drop_duplicates("tract_id").copy()
    cases["match_year"] = 2024
    cases["state"] = cases["tract_id"].map(state_lookup)
    return cases


def _before_cases(
    sites: pd.DataFrame, long_df: pd.DataFrame, before_after: pd.DataFrame, state_lookup: pd.Series
) -> pd.DataFrame:
    rq1_tracts = sites.loc[sites["rq1_eligible"], "tract_id"].drop_duplicates()
    first_build = long_df.drop_duplicates("tract_id").set_index("tract_id")["first_build_year"]
    pre_year = (first_build.loc[rq1_tracts] - 1).clip(lower=2010).astype(int)

    # The 44 before/after tracts use the teammate's own pre_year, already
    # reconciled with her change_* columns -- more authoritative than our
    # build_year - 1 derivation for that subset.
    override = before_after.set_index("tract_id")["pre_year"]
    pre_year.update(override.reindex(pre_year.index).dropna().astype(int))

    rows = []
    for tract_id, match_year in pre_year.items():
        row = long_df[(long_df["tract_id"] == tract_id) & (long_df["acs_year"] == match_year)]
        if row.empty:
            print(f"  WARNING: no ACS row for {tract_id} at match_year={match_year}, skipping")
            continue
        rows.append(row.iloc[0])
    cases = pd.DataFrame(rows).reset_index(drop=True)
    cases["match_year"] = cases["tract_id"].map(pre_year)
    cases["state"] = cases["tract_id"].map(state_lookup)
    return cases


def _enrich_before_after_changes(
    analysis: pd.DataFrame, matches: pd.DataFrame, pool: pd.DataFrame, before_after: pd.DataFrame
) -> pd.DataFrame:
    """Add change_* columns (post[2024] - pre) for the 44 rq4_eligible cases and
    their matched controls, mirroring dc_tract_before_after.csv for the case side."""
    rq4_tracts = set(before_after["tract_id"])
    change_cols = {f"change_{c}": pd.Series(dtype="float64") for c in CHANGE_BASE_COLS}
    analysis = analysis.assign(**change_cols)

    pool_2024 = pool[pool["acs_year"] == 2024].set_index("tract_id")

    # Case rows: take dc_tract_before_after.csv's own change_* columns directly.
    ba_indexed = before_after.set_index("tract_id")
    for base in CHANGE_BASE_COLS:
        col = f"change_{base}"
        if col in ba_indexed.columns:
            is_case_rq4 = analysis["is_case"] & analysis["tract_id"].isin(rq4_tracts)
            analysis.loc[is_case_rq4, col] = analysis.loc[is_case_rq4, "tract_id"].map(ba_indexed[col])

    # Control rows: compute change_* ourselves from each control's pre (already
    # in `analysis`, since this is the before-purpose table) and 2024 snapshot.
    # Match on (tract_id, match_year), not tract_id alone -- the same physical
    # tract can also be a control for a *different*, non-44 case at a different
    # match_year, and that row must NOT get change_* filled in.
    rq4_case_tracts = matches["case_tract_id"].isin(rq4_tracts)
    rq4_control_pairs = set(
        matches.loc[rq4_case_tracts, ["control_tract_id", "match_year"]].itertuples(index=False, name=None)
    )
    pair_keys = list(zip(analysis["tract_id"], analysis["match_year"]))
    is_control_rq4 = ~analysis["is_case"] & pd.Series(
        [key in rq4_control_pairs for key in pair_keys], index=analysis.index
    )
    for base in CHANGE_BASE_COLS:
        post_vals = analysis.loc[is_control_rq4, "tract_id"].map(pool_2024[base])
        analysis.loc[is_control_rq4, f"change_{base}"] = post_vals.values - analysis.loc[is_control_rq4, base].values

    return analysis


def main() -> None:
    cfg = load_config()
    k = cfg["sampling"]["controls_per_case"]
    SAMPLING_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading teammates' Census/RUCA tables...")
    sites = pd.read_csv(DC_CENSUS_DIR / "dc_sites.csv", dtype={"tract_id": str})
    long_df = pd.read_csv(DC_CENSUS_DIR / "dc_tract_acs_long.csv", dtype={"tract_id": str})
    before_after = pd.read_csv(DC_CENSUS_DIR / "dc_tract_before_after.csv", dtype={"tract_id": str})
    state_lookup = _build_state_lookup()
    pool = _load_control_pool(state_lookup)
    print(f"  control pool: {pool['tract_id'].nunique()} tracts that never had a data center")

    print("Matching controls for the cumulative (~1,800) set...")
    cumulative_cases = _cumulative_cases(long_df, state_lookup)
    cumulative_matches = _run_matching(cumulative_cases, pool, k)
    cumulative_analysis = _build_analysis_table(cumulative_cases, cumulative_matches, pool)
    print(f"  {len(cumulative_cases)} cases, {cumulative_matches['relaxed_match'].sum()} relaxed matches")

    print("Matching controls for the before (203) set...")
    before_cases = _before_cases(sites, long_df, before_after, state_lookup)
    before_matches = _run_matching(before_cases, pool, k)
    before_analysis = _build_analysis_table(before_cases, before_matches, pool)
    print(f"  {len(before_cases)} cases, {before_matches['relaxed_match'].sum()} relaxed matches")

    print("Enriching the 44 before/after cases and their controls with change_* columns...")
    before_analysis = _enrich_before_after_changes(before_analysis, before_matches, pool, before_after)

    cumulative_matches.to_csv(SAMPLING_DIR / "cumulative_matches.csv", index=False)
    cumulative_analysis.to_csv(SAMPLING_DIR / "cumulative_analysis.csv", index=False)
    before_matches.to_csv(SAMPLING_DIR / "before_matches.csv", index=False)
    before_analysis.to_csv(SAMPLING_DIR / "before_analysis.csv", index=False)
    print(f"Wrote 4 files to {SAMPLING_DIR}")


if __name__ == "__main__":
    main()
