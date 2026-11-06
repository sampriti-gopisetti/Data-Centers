"""Build the cleaned ACS + RUCA tract tables once, and load them afterwards.

Build (needs CENSUS_API_KEY in .env, plus the two raw downloads in data/raw/):
    python -m src.features.acs_ruca

Load (no API, no raw files needed, just the committed/shared parquet files):
    from src.features.acs_ruca import load_acs_ruca
    df = load_acs_ruca()            # one row per (tract_id, acs_year), RUCA columns attached
"""
import os
from pathlib import Path

import pandas as pd

from . import census, checks, rural_urban

RAW = Path("data/raw")
OUT = Path("data/interim")
XW_PATH = RAW / "tab20_tract20_tract10_natl.txt"
RUCA_PATH = RAW / "RUCA-codes-2020-tract.csv"
ACS_FILE, RUCA_FILE, QC_FILE = "acs_tract_year.parquet", "ruca_tract.parquet", "acs_qc_by_year.csv"

# Columns kept in the saved file (raw counts stay in the local data/raw/acs cache).
KEEP = ["tract_id", "acs_year", "crosswalked", "low_pop_flag",
        "pop_total", "households", "median_hh_income", "median_home_value",
        "poverty_rate", "pct_nh_white", "pct_nonwhite", "pct_nh_black", "pct_nh_aian",
        "pct_nh_asian", "pct_hispanic", "pct_bachelors_plus", "unemployment_rate",
        "pop_density_km2", "income_pctile_year", "home_value_pctile_year",
        "race_total", "nh_white", "nh_black", "nh_aian", "nh_asian", "hispanic", "pov_universe", "pov_below"]


def _slim(acs: pd.DataFrame) -> pd.DataFrame:
    s = acs[KEEP].copy()
    for c in s.select_dtypes("float64").columns:
        s[c] = s[c].round(4 if (s[c].abs().max() <= 1.0001) else 2).astype("float32")
    s["acs_year"] = s["acs_year"].astype("int16")
    return s


def build_all(key, years=range(2010, 2025), xw_path=XW_PATH, ruca_path=RUCA_PATH, out_dir=OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acs = census.build_census_table(years, key, str(xw_path))
    ruca = rural_urban.build_ruca_table(str(ruca_path))
    if "pop_conserved_ratio" not in acs:
        acs["pop_conserved_ratio"] = float("nan")

    qc = acs.groupby("acs_year").agg(
        n_tracts=("tract_id", "size"), pop_total=("pop_total", "sum"),
        pop_conserved_ratio=("pop_conserved_ratio", "first"),
        crosswalked=("crosswalked", "first")).reset_index()
    qc.to_csv(out_dir / QC_FILE, index=False)

    _slim(acs).to_parquet(out_dir / ACS_FILE, index=False, compression="zstd")
    ruca.to_parquet(out_dir / RUCA_FILE, index=False, compression="zstd")

    xw = census.load_crosswalk(str(xw_path))
    checks.basic_report(acs, ruca, set(xw["GEOID_TRACT_20"]))
    for f in (ACS_FILE, RUCA_FILE, QC_FILE):
        print(f"{f}: {(out_dir / f).stat().st_size / 1e6:.1f} MB")
    return acs, ruca


def load_acs_ruca(directory=OUT) -> pd.DataFrame:
    """The ready-to-use table: ACS panel with the static 2020 RUCA columns attached."""
    d = Path(directory)
    acs = pd.read_parquet(d / ACS_FILE)
    ruca = pd.read_parquet(d / RUCA_FILE)
    return acs.merge(ruca, on="tract_id", how="left", validate="many_to_one")


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    build_all(os.environ["CENSUS_API_KEY"])


if __name__ == "__main__":
    main()