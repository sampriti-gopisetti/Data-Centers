"""Entry point for the 'ingest' stage. Run with: python -m src.ingest.run

Pulls each raw source into data/raw/, then flattens it into a tidy interim
table under data/interim/:

  - Compute Atlas: fetched over HTTP (no auth required) and filtered to
    facilityType == "data_center".
  - PNNL IM3 Data Center Atlas: has no public download API — its DOI record
    only serves a placeholder file. Read from a GeoPackage/CSV the user has
    downloaded by hand into data/raw/pnnl/ (see PNNL_MISSING_MESSAGE below).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import DBSCAN

from src.utils import ROOT, load_config

COMPUTE_ATLAS_FACILITIES_URL = (
    "https://raw.githubusercontent.com/ek33450505/compute-atlas/main/data/facilities.json"
)
COMPUTE_ATLAS_META_URL = (
    "https://raw.githubusercontent.com/ek33450505/compute-atlas/main/data/facilities.meta.json"
)

PNNL_DOWNLOAD_URL = "https://doi.org/10.57931/3017294"
PNNL_LAYERS = ("point", "building", "campus")
EARTH_RADIUS_M = 6_371_000

CENSUS_TRACT_BOUNDARY_URL_TMPL = (
    "https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_tract_500k.zip"
)

EPOCH_AI_DATA_CENTERS_URL = "https://epoch.ai/data/data_centers/data_centers.csv"
EPOCH_AI_TIMELINES_URL = "https://epoch.ai/data/data_centers/data_center_timelines.csv"

PNNL_MISSING_MESSAGE = f"""
No PNNL IM3 Data Center Atlas file found under data/raw/pnnl/.

The PNNL dataset has no public download API — only a placeholder file is
served by its DOI record, so it has to be downloaded by hand once:

  1. Visit {PNNL_DOWNLOAD_URL}
  2. Click "Download" and choose GeoPackage (.gpkg) or CSV
  3. Save the file anywhere under: data/raw/pnnl/
  4. Re-run: python -m src.ingest.run
""".strip()


def _fetch_compute_atlas(raw_dir: Path) -> list[dict]:
    out_dir = raw_dir / "compute_atlas"
    out_dir.mkdir(parents=True, exist_ok=True)

    facilities = requests.get(COMPUTE_ATLAS_FACILITIES_URL, timeout=60)
    facilities.raise_for_status()
    (out_dir / "facilities.json").write_bytes(facilities.content)

    meta = requests.get(COMPUTE_ATLAS_META_URL, timeout=60)
    meta.raise_for_status()
    (out_dir / "facilities.meta.json").write_bytes(meta.content)

    return json.loads(facilities.text)


def _flatten_compute_atlas(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        if r.get("facilityType") != "data_center":
            continue
        loc = r.get("location") or {}
        rows.append(
            {
                "compute_atlas_id": r.get("id"),
                "name": r.get("name"),
                "operator": r.get("operator"),
                "status": r.get("status"),
                "confidence": r.get("confidence"),
                "latitude": loc.get("lat"),
                "longitude": loc.get("lon"),
                "city": loc.get("city"),
                "county": loc.get("county"),
                "state": loc.get("state"),
            }
        )
    df = pd.DataFrame(rows)
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def _find_pnnl_source(raw_dir: Path) -> Path:
    pnnl_dir = raw_dir / "pnnl"
    if pnnl_dir.exists():
        gpkg = sorted(pnnl_dir.glob("*.gpkg"))
        if gpkg:
            return gpkg[0]
        csv = sorted(pnnl_dir.glob("*.csv"))
        if csv:
            return csv[0]
    raise FileNotFoundError(PNNL_MISSING_MESSAGE)


def _flatten_pnnl_geopackage(path: Path) -> pd.DataFrame:
    import geopandas as gpd

    frames = []
    for layer in PNNL_LAYERS:
        try:
            gdf = gpd.read_file(path, layer=layer)
        except Exception:  # layer not present in this export
            continue
        gdf = gdf.to_crs(epsg=4326)
        gdf["pnnl_geometry_type"] = layer
        if "lat" in gdf.columns and "lon" in gdf.columns:
            gdf["latitude"] = gdf["lat"]
            gdf["longitude"] = gdf["lon"]
        else:
            # Equal-area projection (CONUS Albers) so the centroid isn't
            # distorted by computing it directly in lat/lon degrees.
            centroid = gdf.geometry.to_crs(epsg=5070).centroid.to_crs(epsg=4326)
            gdf["latitude"] = centroid.y
            gdf["longitude"] = centroid.x
        frames.append(pd.DataFrame(gdf.drop(columns="geometry")))
    if not frames:
        raise ValueError(f"None of {PNNL_LAYERS} found as layers in {path}")
    return pd.concat(frames, ignore_index=True)


def _flatten_pnnl_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pnnl_geometry_type"] = df["type"] if "type" in df.columns else "unknown"
    df["latitude"] = df["lat"]
    df["longitude"] = df["lon"]
    return df


def _collapse_pnnl_sites(raw: pd.DataFrame, dedup_distance_m: float) -> pd.DataFrame:
    """Collapse PNNL's point/building/campus rows into one row per physical
    site. The source's own usage notes say a site can appear as overlapping
    point/building/campus geometries, and campus rows can duplicate across a
    county line — so rows within dedup_distance_m of each other are treated
    as the same site."""
    raw = raw.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    coords = np.radians(raw[["latitude", "longitude"]].to_numpy())
    eps = dedup_distance_m / EARTH_RADIUS_M  # meters -> radians on the unit sphere
    labels = DBSCAN(eps=eps, min_samples=1, metric="haversine").fit_predict(coords)
    raw = raw.assign(_group=labels)

    id_col = "id" if "id" in raw.columns else "ref"
    rows = []
    for _, group in raw.groupby("_group"):
        rep = group.iloc[0]
        point_rows = group[group["pnnl_geometry_type"] == "point"]
        anchor = point_rows.iloc[0] if len(point_rows) else rep
        rows.append(
            {
                "pnnl_id": str(rep[id_col]),
                "name": rep.get("name"),
                "operator": rep.get("operator"),
                "state": rep.get("state_abb", rep.get("state")),
                "county": rep.get("county"),
                "latitude": anchor["latitude"],
                "longitude": anchor["longitude"],
                "pnnl_record_count": len(group),
                "pnnl_geometry_types": ",".join(sorted(group["pnnl_geometry_type"].unique())),
                "pnnl_ids": ",".join(sorted(group[id_col].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def _fetch_census_tracts(raw_dir: Path, tract_vintage: int) -> Path:
    """Download the Census Bureau's national cartographic-boundary tract
    file (one zip, all states) for the configured vintage year."""
    import zipfile

    out_dir = raw_dir / "census_tracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"cb_{tract_vintage}_us_tract_500k.zip"

    if not zip_path.exists():
        url = CENSUS_TRACT_BOUNDARY_URL_TMPL.format(year=tract_vintage)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    shp_files = list(out_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found after extracting {zip_path}")
    return shp_files[0]


def _flatten_census_tracts(shp_path: Path):
    import geopandas as gpd

    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    keep = ["GEOID", "STATEFP", "COUNTYFP", "geometry"]
    return gdf[[c for c in keep if c in gdf.columns]]


def _fetch_epoch_ai(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = raw_dir / "epoch_ai"
    out_dir.mkdir(parents=True, exist_ok=True)

    centers = requests.get(EPOCH_AI_DATA_CENTERS_URL, timeout=60)
    centers.raise_for_status()
    (out_dir / "data_centers.csv").write_bytes(centers.content)

    timelines = requests.get(EPOCH_AI_TIMELINES_URL, timeout=60)
    timelines.raise_for_status()
    (out_dir / "data_center_timelines.csv").write_bytes(timelines.content)

    from io import StringIO

    return pd.read_csv(StringIO(centers.text)), pd.read_csv(StringIO(timelines.text))


def _flatten_epoch_ai(centers: pd.DataFrame, timelines: pd.DataFrame) -> pd.DataFrame:
    """One row per Epoch AI data center: name, owner/users (for matching),
    and the year of the first timeline entry where a building is reported
    operational — our best proxy for "construction year" since Epoch AI
    doesn't publish a single clean construction-date field."""
    operational = timelines[timelines["Buildings operational"] > 0].copy()
    operational["year"] = pd.to_datetime(operational["Date"]).dt.year
    first_operational = (
        operational.sort_values("Date").groupby("Data center", as_index=False).first()[
            ["Data center", "year"]
        ]
    )

    import re

    confidence_tag_re = re.compile(r"\s*#(confident|likely|speculative)\b", re.I)
    strip_tags = lambda s: confidence_tag_re.sub("", s) if isinstance(s, str) else s

    centers = centers.rename(columns={"Name": "Data center"})
    centers = centers.assign(
        Owner=centers["Owner"].map(strip_tags), Users=centers["Users"].map(strip_tags)
    )
    merged = centers[["Data center", "Owner", "Users"]].merge(
        first_operational, on="Data center", how="inner"
    )
    return merged.rename(
        columns={
            "Data center": "name",
            "Owner": "owner",
            "Users": "users",
            "year": "epoch_operational_year",
        }
    )


def main() -> None:
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    interim_dir = ROOT / cfg["paths"]["interim"]
    interim_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching Compute Atlas facilities...")
    records = _fetch_compute_atlas(raw_dir)
    compute_df = _flatten_compute_atlas(records)
    compute_out = interim_dir / "compute_atlas_data_centers.parquet"
    compute_df.to_parquet(compute_out, index=False)
    print(f"  {len(compute_df)} Compute Atlas data_center records -> {compute_out}")

    print("Looking for PNNL source file...")
    pnnl_path = _find_pnnl_source(raw_dir)
    print(f"  found {pnnl_path}")
    if pnnl_path.suffix == ".gpkg":
        pnnl_raw = _flatten_pnnl_geopackage(pnnl_path)
    else:
        pnnl_raw = _flatten_pnnl_csv(pnnl_path)

    dedup_distance_m = cfg["registry"]["dedup_distance_m"]
    pnnl_sites = _collapse_pnnl_sites(pnnl_raw, dedup_distance_m)
    pnnl_out = interim_dir / "pnnl_sites.parquet"
    pnnl_sites.to_parquet(pnnl_out, index=False)
    print(
        f"  {len(pnnl_raw)} raw PNNL rows -> {len(pnnl_sites)} collapsed sites -> {pnnl_out}"
    )

    print("Fetching Census tract boundaries...")
    tract_vintage = cfg["tract_vintage"]
    shp_path = _fetch_census_tracts(raw_dir, tract_vintage)
    tracts = _flatten_census_tracts(shp_path)
    tracts_out = interim_dir / "census_tracts.parquet"
    tracts.to_parquet(tracts_out)
    print(f"  {len(tracts)} tracts ({tract_vintage} vintage) -> {tracts_out}")

    print("Fetching Epoch AI data centers...")
    centers, timelines = _fetch_epoch_ai(raw_dir)
    epoch_df = _flatten_epoch_ai(centers, timelines)
    epoch_out = interim_dir / "epoch_ai_construction_dates.parquet"
    epoch_df.to_parquet(epoch_out, index=False)
    print(f"  {len(epoch_df)} Epoch AI sites with a resolvable operational year -> {epoch_out}")


if __name__ == "__main__":
    main()
