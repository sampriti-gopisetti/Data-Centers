"""ACS 5-year tract features, harmonized to 2020 tract boundaries.

Output: one row per (tract_id, acs_year) with raw counts and derived rates.
tract_id is always the 11-digit 2020 tract GEOID, stored as a string.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# 50 states + DC (Puerto Rico excluded; add "72" if your data center sources include it)
STATES = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16",
          "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
          "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
          "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56"]

CORE_VARS = {
    "B01003_001E": "pop_total",
    "B19013_001E": "median_hh_income",    # median: needs weighted averaging in crosswalk
    "B25077_001E": "median_home_value",   # median: owner-occupied units
    "B17001_001E": "pov_universe",
    "B17001_002E": "pov_below",
    "B03002_001E": "race_total",
    "B03002_003E": "nh_white",
    "B03002_004E": "nh_black",
    "B03002_005E": "nh_aian",
    "B03002_006E": "nh_asian",
    "B03002_012E": "hispanic",
    "B11001_001E": "households",
    "B25003_002E": "owner_units",
}
# Optional: if the Census API rejects them for an early year, that year gets NaN for these.
OPTIONAL_VARS = {
    "B15003_001E": "edu_universe",   # population 25+
    "B15003_022E": "_bach",          # bachelor's
    "B15003_023E": "_mast",          # master's
    "B15003_024E": "_prof",          # professional degree
    "B15003_025E": "_doct",          # doctorate
    "B23025_003E": "labor_force",    # civilian labor force
    "B23025_005E": "unemployed",
}
VARS = {**CORE_VARS, **OPTIONAL_VARS}
COUNT_COLS = ["pop_total", "pov_universe", "pov_below", "race_total", "nh_white",
              "nh_black", "nh_aian", "nh_asian", "hispanic", "households", "owner_units",
              "edu_universe", "bach_plus", "labor_force", "unemployed"]
MEDIAN_WEIGHTS = {"median_hh_income": "households", "median_home_value": "owner_units"}
SENTINEL_MAX = -222222222  # Census uses -666666666, -999999999, etc. for "not available"
CACHE_VERSION = "v2"       # bump if the variable list changes, so stale caches are ignored


def fetch_state(year: int, state: str, key: str, variables: list, retries: int = 3) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": ",".join(variables), "for": "tract:*", "in": f"state:{state} county:*", "key": key}
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            try:
                rows = r.json()
            except ValueError:
                raise RuntimeError(f"ACS {year} state {state}: non-JSON reply (bad API key?): {r.text[:200]}")
            return pd.DataFrame(rows[1:], columns=rows[0])
        if r.status_code == 204:
            return pd.DataFrame()
        if r.status_code == 400:
            raise ValueError(f"ACS {year}: variable not available ({r.text[:150]})")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ACS {year} state {state}: HTTP {r.status_code} {r.text[:200]}")


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Rename, make numeric, null out Census sentinel codes, build 11-digit tract id."""
    out = pd.DataFrame({"tract_id": df["state"] + df["county"] + df["tract"]})
    for code, name in VARS.items():
        if code in df.columns:
            s = pd.to_numeric(df[code], errors="coerce")
            out[name] = s.mask(s <= SENTINEL_MAX)
        else:
            out[name] = np.nan
    out["bach_plus"] = out[["_bach", "_mast", "_prof", "_doct"]].sum(axis=1, min_count=4)
    return out.drop(columns=["_bach", "_mast", "_prof", "_doct"])


def pull_year(year: int, key: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"acs5_tract_{year}_{CACHE_VERSION}.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"tract_id": str})
    variables = list(VARS)
    try:
        first = fetch_state(year, STATES[0], key, variables)
    except ValueError as e:
        print(f"[ACS {year}] optional variables unavailable, using core only ({e})")
        variables = list(CORE_VARS)
        first = fetch_state(year, STATES[0], key, variables)  # raises if core vars fail too
    frames = [first] + [fetch_state(year, s, key, variables) for s in STATES[1:]]
    out = clean_raw(pd.concat([x for x in frames if len(x)], ignore_index=True))
    out.to_csv(f, index=False)
    print(f"[ACS {year}] pulled {len(out)} tracts")
    return out


def load_crosswalk(path: str) -> pd.DataFrame:
    """Census 2010->2020 tract relationship file (pipe-delimited). Check column names with
    print(pd.read_csv(path, sep='|', nrows=2).columns) if this raises a KeyError."""
    x = pd.read_csv(path, sep="|", dtype=str)
    for c in ["AREALAND_PART", "AREALAND_TRACT_10", "AREALAND_TRACT_20"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[x["GEOID_TRACT_10"].notna() & x["GEOID_TRACT_20"].notna()]
    x = x[x["AREALAND_TRACT_10"] > 0].copy()
    x["w"] = x["AREALAND_PART"] / x["AREALAND_TRACT_10"]
    x["w"] = x["w"] / x.groupby("GEOID_TRACT_10")["w"].transform("sum")  # shares sum to 1
    return x[["GEOID_TRACT_10", "GEOID_TRACT_20", "w", "AREALAND_TRACT_20"]]


def to_2020_tracts(df10: pd.DataFrame, xw: pd.DataFrame) -> pd.DataFrame:
    """Area-weighted 2010 -> 2020 reallocation. Counts are split; medians are
    re-averaged using households / owner units as weights."""
    m = df10.merge(xw, left_on="tract_id", right_on="GEOID_TRACT_10", how="inner")
    out = m[COUNT_COLS].mul(m["w"], axis=0).groupby(m["GEOID_TRACT_20"]).sum(min_count=1)
    for med, wcol in MEDIAN_WEIGHTS.items():
        wt = m[wcol] * m["w"]
        ok = m[med].notna() & wt.notna() & (wt > 0)
        num = (m[med] * wt).where(ok).groupby(m["GEOID_TRACT_20"]).sum(min_count=1)
        den = wt.where(ok).groupby(m["GEOID_TRACT_20"]).sum(min_count=1)
        out[med] = num / den
    out = out.rename_axis("tract_id").reset_index()
    out["pop_conserved_ratio"] = out["pop_total"].sum() / df10["pop_total"].sum()  # should be ~1
    return out


def add_derived(df: pd.DataFrame, land_km2: pd.Series) -> pd.DataFrame:
    """Derived rates, density, and within-year percentile ranks (so income and home value are
    comparable across years without needing a CPI adjustment)."""
    d = df.copy()
    safe = lambda n, den: (d[n] / d[den]).where(d[den] > 0)
    d["pct_nh_white"] = safe("nh_white", "race_total")
    d["pct_nonwhite"] = 1 - d["pct_nh_white"]
    d["pct_nh_black"] = safe("nh_black", "race_total")
    d["pct_nh_aian"] = safe("nh_aian", "race_total")
    d["pct_nh_asian"] = safe("nh_asian", "race_total")
    d["pct_hispanic"] = safe("hispanic", "race_total")
    d["poverty_rate"] = safe("pov_below", "pov_universe")
    d["pct_bachelors_plus"] = safe("bach_plus", "edu_universe")
    d["unemployment_rate"] = safe("unemployed", "labor_force")
    d["land_km2"] = d["tract_id"].map(land_km2)
    d["pop_density_km2"] = (d["pop_total"] / d["land_km2"]).where(d["land_km2"] > 0)
    d["income_pctile_year"] = d["median_hh_income"].rank(pct=True)
    d["home_value_pctile_year"] = d["median_home_value"].rank(pct=True)
    d["low_pop_flag"] = d["pop_total"].fillna(0) < 50  # ratios unreliable / undefined here
    return d


def build_census_table(years, key, xwalk_path, cache_dir="data/raw/acs") -> pd.DataFrame:
    xw = load_crosswalk(xwalk_path)
    land = xw.drop_duplicates("GEOID_TRACT_20").set_index("GEOID_TRACT_20")["AREALAND_TRACT_20"] / 1e6
    frames = []
    for y in years:
        raw = pull_year(y, key, Path(cache_dir))
        native = y >= 2020  # ACS 5-year releases from 2020 on are on 2020 tract boundaries
        t = raw if native else to_2020_tracts(raw, xw)
        t = add_derived(t, land)
        t["acs_year"] = y
        t["crosswalked"] = not native
        frames.append(t)
    return pd.concat(frames, ignore_index=True)