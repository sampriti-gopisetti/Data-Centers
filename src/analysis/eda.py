"""Exploratory analysis of the Step 2 model-ready dataset. Run with:
python -m src.analysis.eda

Computes the summary statistics behind the midterm report's Data section:
class balance, case-vs-control feature comparisons, geographic distribution
by state, and correlation of each feature with is_case. Reads
data/processed/modeling/cumulative_modeling.csv (the lead dataset -- current
day, all data-center tracts + matched controls). Prints everything to stdout
and writes data/processed/modeling/eda_summary.json so the numbers used in
the report are reproducible and checkable against the source data.
"""
from __future__ import annotations

import json

import pandas as pd

from src.utils import ROOT

MODELING_DIR = ROOT / "data" / "processed" / "modeling"
REFERENCE_DIR = ROOT / "data" / "processed" / "reference"

COMPARISON_FEATURES = ["median_hh_income", "poverty_rate", "pop_density_km2", "pct_developed"]

NUMERIC_FEATURES_FOR_CORRELATION = [
    "pop_total", "households", "median_hh_income", "median_home_value", "poverty_rate",
    "pct_nonwhite", "pct_nh_black", "pct_hispanic", "pct_nh_asian", "pct_bachelors_plus",
    "unemployment_rate", "pop_density_km2", "pct_developed", "pct_forest", "pct_agriculture",
    "ej_pm25", "ej_ozone", "ej_dslpm", "ej_no2", "ej_demogidx_2", "tavg_norm_F", "pcp_norm_in",
]


def _tract_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(11)


def _state_lookup() -> pd.Series:
    ej = pd.read_csv(
        REFERENCE_DIR / "ejscreen_tract_clean_50states_dc.csv",
        dtype={"GEOID": str},
        usecols=["GEOID", "state_abbr"],
    )
    ej["GEOID"] = _tract_str(ej["GEOID"])
    return ej.set_index("GEOID")["state_abbr"]


def main() -> None:
    df = pd.read_csv(MODELING_DIR / "cumulative_modeling.csv", dtype={"tract_id": str})
    df["tract_id"] = _tract_str(df["tract_id"])

    summary: dict = {}

    class_balance = df["is_case"].value_counts().to_dict()
    summary["class_balance"] = {"case": int(class_balance.get(True, 0)), "control": int(class_balance.get(False, 0))}
    print("Class balance:", summary["class_balance"])

    medians = df.groupby("is_case")[COMPARISON_FEATURES].median()
    summary["feature_medians_by_group"] = {
        "case": medians.loc[True].round(4).to_dict(),
        "control": medians.loc[False].round(4).to_dict(),
    }
    print("\nMedian feature values, case vs control:")
    print(medians.round(3))

    state_lookup = _state_lookup()
    df["state"] = df["tract_id"].map(state_lookup)
    state_counts = (
        df[df["is_case"]].groupby("state").size().sort_values(ascending=False).head(10)
    )
    summary["top_states_by_case_count"] = state_counts.to_dict()
    print("\nTop 10 states by data-center tract count:")
    print(state_counts)

    numeric = df[NUMERIC_FEATURES_FOR_CORRELATION + ["is_case"]].copy()
    numeric["is_case"] = numeric["is_case"].astype(int)
    corr_with_case = numeric.corr(numeric_only=True)["is_case"].drop("is_case")
    top_corr = corr_with_case.reindex(corr_with_case.abs().sort_values(ascending=False).index).head(10)
    summary["top_correlations_with_is_case"] = top_corr.round(4).to_dict()
    print("\nTop 10 features by |correlation| with is_case:")
    print(top_corr.round(4))

    full_corr = numeric.corr(numeric_only=True)
    full_corr.to_csv(MODELING_DIR / "feature_correlation_matrix.csv")
    print(f"\nFull correlation matrix written to {MODELING_DIR / 'feature_correlation_matrix.csv'}")

    missing_pct = (df.isna().mean() * 100).round(2)
    summary["missing_pct_nonzero"] = missing_pct[missing_pct > 0].sort_values(ascending=False).to_dict()

    with open(MODELING_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {MODELING_DIR / 'eda_summary.json'}")


if __name__ == "__main__":
    main()
