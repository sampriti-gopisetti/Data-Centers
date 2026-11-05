"""Export Census/RUCA data linked to each data center, organised by the four research questions.

Save as src/features/export_dc_data.py. Needs the FULL 2010-2024 panel (python -m src.features.acs_ruca).
Run from the repo root:
    python -m src.features.export_dc_data data/raw/dc_registry.csv

Writes data/processed/dc_census/ (small CSVs you can send; README_dc_tables.md explains each column).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .acs_ruca import load_acs_ruca
from .dc_table import TERRITORIES, load_registry

FIRST_ACS, LAST_ACS = 2010, 2024
COUNTS = ["pop_total", "households", "pov_below", "pov_universe", "race_total",
          "nh_white", "nh_black", "nh_aian", "nh_asian", "hispanic"]
CHANGE_VARS = ["pop_total", "pct_nonwhite", "pct_nh_black", "pct_hispanic", "pct_nh_asian", "pct_nh_aian",
               "poverty_rate", "median_hh_income", "income_pctile_year", "median_home_value",
               "home_value_pctile_year", "pct_bachelors_plus", "unemployment_rate", "pop_density_km2"]
RUCA_COLS = ["ruca_primary", "ruca_class", "is_rural"]  # static 2020 RUCA codes, same value in every year
SITE_COLS = ["site_id", "name", "operator", "state", "county", "tract_id", "county_fips", "latitude", "longitude",
             "status", "construction_year_confidence", "construction_year_source", "year_known", "build_year",
             "start_year_basis", "exposure_start_year", "tract_n_sites", "tract_has_undated_site",
             "rq1_eligible", "rq2_eligible", "rq3_eligible", "rq4_eligible"]


def build_sites(path):
    d = load_registry(path)
    d = d[d.tract_id.notna() & ~d.state.isin(TERRITORIES)].copy()
    d["status"] = d["operational_status"].fillna("pnnl_only")        # 'pnnl_only' = no Compute Atlas status
    d = d[d.status != "cancelled"]
    d["county_fips"] = d["tract_id"].str[:5]
    # a build year is trusted only if NLCD detected the change or Epoch AI supplied it
    d["year_known"] = d.construction_year_confidence.isin(["high", "low"]) | (d.construction_year_source == "epoch_ai")
    d["build_year"] = d["construction_year"].where(d["year_known"])
    g = d.groupby("tract_id")
    d["tract_n_sites"] = g["site_id"].transform("size")
    d["tract_has_undated_site"] = g["year_known"].transform(lambda s: bool((~s).any()))
    op = d["status"] == "operational"
    d["start_year_basis"] = np.where(~d.year_known, "unknown",
                                     np.where(d.build_year < FIRST_ACS, "pre-2010: start at 2010", "known"))
    d["exposure_start_year"] = np.where(d.year_known, d.build_year.clip(lower=FIRST_ACS), np.nan)
    # RQ1 disproportionate targeting: dated, built 2011-2025, so a pre-construction ACS year exists
    d["rq1_eligible"] = d.year_known & d.build_year.between(FIRST_ACS + 1, 2025)
    # RQ2 current snapshot: operational today
    d["rq2_eligible"] = op
    # RQ3 cumulative exposure: operational with a known build year (pre-2010 counted from 2010)
    d["rq3_eligible"] = op & d.year_known & (d.build_year <= LAST_ACS)
    # RQ4 before/after: operational, built 2011-2019 (ACS 2024 covers 2020-2024, fully after the build)
    d["rq4_eligible"] = op & d.year_known & d.build_year.between(FIRST_ACS + 1, LAST_ACS - 5)
    return d[SITE_COLS].reset_index(drop=True)


def tract_summary(sites):
    t = sites.groupby("tract_id").agg(
        n_sites=("site_id", "size"), n_operational=("status", lambda s: int((s == "operational").sum())),
        n_dated=("year_known", "sum"), first_build_year=("build_year", "min")).reset_index()
    t["build_year_trusted"] = t["first_build_year"].where(t["n_dated"] == t["n_sites"])   # all sites dated
    return t


def tract_long(panel, sites):
    """Every ACS year for every data center tract, with timing columns for before/after work."""
    t = tract_summary(sites)
    out = panel.merge(t, on="tract_id", how="inner")
    out["years_since_build"] = out["acs_year"] - out["build_year_trusted"]
    out["window_before_build"] = out["acs_year"] < out["build_year_trusted"]          # window ends before the build year
    out["window_after_build"] = (out["acs_year"] - 4) > out["build_year_trusted"]    # window starts after the build year
    return out.sort_values(["tract_id", "acs_year"]).reset_index(drop=True)


def site_today(panel, sites):
    """RQ2: one row per operational site with its tract's most recent ACS profile."""
    today = panel[panel.acs_year == LAST_ACS]
    s = sites[sites.rq2_eligible]
    return s.merge(today, on="tract_id", how="left")


def tract_exposure(panel, sites, assume_2010=False):
    """RQ3: people counted once per tract. Person-years = sum of yearly counts from the start year to 2024.

    assume_2010=True also includes operational sites with an UNKNOWN build year, counting them from 2010
    (an upper bound: some of them may have been built or retrofitted after 2010)."""
    s = sites[sites.status == "operational"].copy() if assume_2010 else sites[sites.rq3_eligible].copy()
    s["start"] = s["exposure_start_year"].fillna(FIRST_ACS) if assume_2010 else s["exposure_start_year"]
    st = s.groupby("tract_id").agg(exposure_start_year=("start", "min"),
                                   n_sites=("site_id", "size"),
                                   start_year_uncertain=("tract_has_undated_site", "max")).reset_index()
    p = panel.merge(st, on="tract_id")
    p = p[(p.acs_year >= p.exposure_start_year) & (p.acs_year <= LAST_ACS)]
    py = p.groupby("tract_id")[COUNTS].sum(min_count=1).add_prefix("person_years_")
    py["n_years"] = p.groupby("tract_id").size()
    today = panel[panel.acs_year == LAST_ACS].set_index("tract_id")[COUNTS].add_prefix("today_")
    return st.merge(py, on="tract_id", how="left").merge(today, on="tract_id", how="left")


def before_after(panel, sites):
    """RQ4: tract profile the year before construction vs ACS 2024, for tracts whose FIRST data center is dated."""
    s = sites[sites.rq4_eligible]
    t = s.groupby("tract_id").agg(build_year=("build_year", "min"), n_sites=("site_id", "size"),
                                  has_undated=("tract_has_undated_site", "max")).reset_index()
    t = t[~t.has_undated].drop(columns="has_undated")        # otherwise the tract may have had a data center earlier
    t["pre_year"], t["post_year"] = (t["build_year"] - 1).astype(int), LAST_ACS
    cols = ["tract_id", "acs_year", "crosswalked"] + CHANGE_VARS
    pre = t.merge(panel[cols], left_on=["tract_id", "pre_year"], right_on=["tract_id", "acs_year"], how="left")
    pre = pre.drop(columns="acs_year").rename(columns={c: f"pre_{c}" for c in CHANGE_VARS + ["crosswalked"]})
    post = panel[panel.acs_year == LAST_ACS][["tract_id"] + CHANGE_VARS].rename(columns={c: f"post_{c}" for c in CHANGE_VARS})
    out = pre.merge(post, on="tract_id", how="left")
    for c in CHANGE_VARS:
        out[f"change_{c}"] = out[f"post_{c}"] - out[f"pre_{c}"]
    return out


README = """# Data-center-linked Census tables

All tables use 2020 census tract IDs (11-digit strings: read tract_id as text). ACS years are 5-year releases
(acs_year 2024 = 2020-2024). 2010-2019 are crosswalked from 2010 to 2020 tracts (approximate). Sites are non-cancelled
records with a tract; territories excluded.

- dc_sites.csv: one row per site, with tract_id and flags rq1..rq4_eligible. build_year is NaN when NLCD could not date the site
  (year_known False). start_year_basis: known / pre-2010 (counted from 2010) / unknown.
- dc_tract_acs_long.csv: every ACS year (2010-2024) for every data center tract. years_since_build, window_before_build
  (ACS window ends before the build year) and window_after_build (starts after it) are for before/after work.
- dc_site_acs_today.csv (RQ2): operational sites with their tract's ACS 2024 profile.
- dc_tract_exposure.csv (RQ3): per tract, person_years_* = sum of yearly counts from exposure_start_year to 2024;
  today_* = counts in ACS 2024. People are counted once per tract. start_year_uncertain = tract also has an undated site.
- dc_tract_exposure_assume2010.csv (RQ3, upper bound): same, but operational sites with an unknown build year are
  counted from 2010 (some may have been built or retrofitted later, so this can overcount).
- dc_tract_before_after.csv (RQ4): pre (year before the build) vs post (ACS 2024) values and changes.
- acs_ruca_all_tracts_2010_2024.csv.gz (LARGE, for negative sampling): the full panel, every US tract and ACS year, with
  has_data_center (tract holds any non-cancelled site) and has_operational_site. Controls = rows with has_data_center False.
RUCA (ruca_primary 1-10, ruca_class, is_rural) is a column in every table. It is the 2020 code, so it is the same in every year.
Caveats: ACS 5-year values are rolling averages with margins of error (not included); crosswalked pre-2020 years add error
exactly where change is measured; 'near' = inside the data center's tract.
"""


def main(registry_path, out_dir="data/processed/dc_census"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = load_acs_ruca()
    sites = build_sites(registry_path)
    print(f"sites: {len(sites)} in {sites.tract_id.nunique()} tracts; tracts missing from ACS panel: "
          f"{(~sites.tract_id.drop_duplicates().isin(set(panel.tract_id))).sum()}")
    tables = {"dc_sites": sites, "dc_tract_acs_long": tract_long(panel, sites), "dc_site_acs_today": site_today(panel, sites),
              "dc_tract_exposure": tract_exposure(panel, sites),
              "dc_tract_exposure_assume2010": tract_exposure(panel, sites, assume_2010=True),
              "dc_tract_before_after": before_after(panel, sites)}
        # RUCA as a feature column in every table (the long/today tables already have it from the panel)
    ruca = panel[["tract_id"] + RUCA_COLS].drop_duplicates("tract_id")
    for name in ["dc_sites", "dc_tract_exposure", "dc_tract_exposure_assume2010", "dc_tract_before_after"]:
        tables[name] = tables[name].merge(ruca, on="tract_id", how="left")
    for name, df in tables.items():
        assert all(c in df.columns for c in RUCA_COLS), f"RUCA missing from {name}"
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df):,} rows, {(out / f'{name}.csv').stat().st_size / 1e6:.1f} MB")
    # the full panel, for teammates drawing negative samples (the tables above only hold data center tracts)
    allp = panel.copy()
    allp["has_data_center"] = allp.tract_id.isin(set(sites.tract_id))
    allp["has_operational_site"] = allp.tract_id.isin(set(sites.loc[sites.status == "operational", "tract_id"]))
    name = "acs_ruca_all_tracts_2010_2024.csv.gz"
    allp.to_csv(out / name, index=False)
    print(f"{name}: {len(allp):,} rows, {(out / name).stat().st_size / 1e6:.1f} MB "
          f"({allp.loc[~allp.has_data_center, 'tract_id'].nunique():,} tracts with no data center)")
    (out / "README_dc_tables.md").write_text(README)
    print("\nEligible sites / tracts by research question:")
    for q in ["rq1", "rq2", "rq3", "rq4"]:
        e = sites[sites[f"{q}_eligible"]]
        print(f"  {q}: {len(e)} sites in {e.tract_id.nunique()} tracts")
    print(f"  RQ1 tracts with a dated site and no undated site: "
          f"{sites[sites.rq1_eligible & ~sites.tract_has_undated_site].tract_id.nunique()}")
    print(f"  RQ3 start year unknown (excluded): {int(((sites.status == 'operational') & ~sites.year_known).sum())} operational sites")
    print(f"  RQ4 tracts after dropping tracts with undated sites: {len(tables['dc_tract_before_after'])}")
    return tables


if __name__ == "__main__":
    main(sys.argv[1])