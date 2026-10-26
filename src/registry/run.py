"""Entry point for the 'registry' stage. Run with: python -m src.registry.run

Builds the deduplicated data center registry — Step 1 of the proposal
("build the data center dataset") — in three parts:

1. Dedup/merge: match PNNL sites (data/interim/pnnl_sites.parquet) against
   Compute Atlas data-center records
   (data/interim/compute_atlas_data_centers.parquet). For each PNNL site,
   every Compute Atlas record within registry.dedup_distance_m is a match
   candidate, scored on name/operator token overlap:
     - exactly one strong-agreement candidate  -> high_confidence_match
     - more than one strong-agreement candidate -> manual_review (multiple_plausible_nearby)
     - only weak-agreement candidates           -> manual_review (weak_or_distant_match)
     - no candidate within range                -> unmatched (no_supported_match)
   Compute Atlas records no PNNL site claims become their own sites
   (compute_only_no_pnnl_candidate).
2. Census tract linkage: point-in-polygon join against
   data/interim/census_tracts.parquet.
3. Construction year: for each site, sample Annual NLCD land cover
   (via Google Earth Engine) at that point for every year in
   [years.nlcd_min_year, years.max_construction] and look for the first
   stable transition into a "Developed" class. Requires GEE_PROJECT in
   .env and a completed `earthengine authenticate`. Then, for the small
   number of sites that match an Epoch AI "Data on AI data centers" entry
   by name/operator (no location data to match on, so held to a stricter
   agreement bar than the PNNL<->Compute Atlas match), Epoch AI's
   satellite-imagery-derived operational year overrides the NLCD estimate
   — it's more precise for exactly the large/complex campuses where a
   single 30m NLCD pixel is least reliable. The override is recorded in
   construction_year_source ("nlcd" vs "epoch_ai"), never silent.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ROOT, load_config, load_dotenv

EARTH_RADIUS_M = 6_371_000
STRONG_AGREEMENT = 0.34  # >=1/3 of combined name+operator tokens shared

NLCD_COLLECTION = "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER"
NLCD_DEVELOPED_CLASSES = {21, 22, 23, 24}  # Open Space, Low/Medium/High Intensity
NLCD_SAMPLE_SCALE_M = 30
NLCD_BATCH_SIZE = 500
NLCD_CALL_TIMEOUT_S = 60  # a getInfo() call can hang indefinitely on a bad connection
NLCD_MAX_RETRIES = 3

# Epoch AI has no coordinates to match on, so it needs a higher agreement bar
# (and a clear margin over the runner-up) than the geo-assisted PNNL<->Compute
# Atlas match, to avoid confusing same-operator sites in different cities.
EPOCH_STRONG_AGREEMENT = 0.5
EPOCH_AMBIGUITY_MARGIN = 0.15

GEE_MISSING_MESSAGE = """
Google Earth Engine is not configured.

Construction-year estimation queries Annual NLCD through GEE, which needs a
one-time setup:
  1. Sign up (free) at https://code.earthengine.google.com and note your
     Cloud project ID.
  2. Run: .venv/bin/earthengine authenticate
  3. Add to a .env file in this repo: GEE_PROJECT=your-project-id
""".strip()

_SUFFIX_RE = re.compile(
    r"\b(inc|llc|corp|corporation|co|company|ltd|data ?centers?|campus|dc\d*)\b", re.I
)
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_tokens(text) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    text = _SUFFIX_RE.sub(" ", text.lower())
    text = _PUNCT_RE.sub(" ", text)
    return {t for t in text.split() if len(t) > 1}


def _name_operator_agreement(a_row: pd.Series, b_row: pd.Series) -> float:
    a = _normalize_tokens(a_row.get("name")) | _normalize_tokens(a_row.get("operator"))
    b = _normalize_tokens(b_row.get("name")) | _normalize_tokens(b_row.get("operator"))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _match_candidates(
    pnnl: pd.DataFrame, compute: pd.DataFrame, max_distance_m: float
) -> dict[int, list[tuple[int, float, float]]]:
    """pnnl index label -> list of (compute index label, distance_m, score)."""
    matches: dict[int, list[tuple[int, float, float]]] = {i: [] for i in pnnl.index}
    pnnl_lat = pnnl["latitude"].to_numpy()
    pnnl_lon = pnnl["longitude"].to_numpy()
    for ci, crow in compute.iterrows():
        d = _haversine_m(pnnl_lat, pnnl_lon, crow["latitude"], crow["longitude"])
        for pi in np.where(d <= max_distance_m)[0]:
            score = _name_operator_agreement(pnnl.iloc[pi], crow)
            matches[pnnl.index[pi]].append((ci, float(d[pi]), score))
    return matches


def _registry_row(
    pnnl_row: pd.Series | None,
    compute_row: pd.Series | None,
    match_status: str,
    review_reason: str | None,
    distance_m: float | None = None,
    score: float | None = None,
) -> dict:
    pick = lambda field: (
        pnnl_row[field] if pnnl_row is not None and pd.notna(pnnl_row.get(field)) else
        (compute_row[field] if compute_row is not None else None)
    )
    pnnl_id = pnnl_row["pnnl_id"] if pnnl_row is not None else None
    compute_atlas_id = compute_row["compute_atlas_id"] if compute_row is not None else None
    site_id = f"PNNL:{pnnl_id}" if pnnl_id is not None else f"CA:{compute_atlas_id}"

    return {
        "site_id": site_id,
        "name": pick("name"),
        "operator": pick("operator"),
        "state": pick("state"),
        "county": pick("county"),
        "latitude": pick("latitude"),
        "longitude": pick("longitude"),
        "operational_status": compute_row["status"] if compute_row is not None else None,
        "pnnl_id": pnnl_id,
        "pnnl_record_count": pnnl_row["pnnl_record_count"] if pnnl_row is not None else None,
        "pnnl_geometry_types": pnnl_row["pnnl_geometry_types"] if pnnl_row is not None else None,
        "compute_atlas_id": compute_atlas_id,
        "match_status": match_status,
        "review_required": match_status == "manual_review",
        "review_reason": review_reason,
        "match_distance_m": distance_m,
        "match_name_operator_score": score,
    }


def _build_registry(pnnl: pd.DataFrame, compute: pd.DataFrame, max_distance_m: float) -> pd.DataFrame:
    matches = _match_candidates(pnnl, compute, max_distance_m)
    matched_compute_ids: set = set()
    rows = []

    for pi, prow in pnnl.iterrows():
        candidates = sorted(matches[pi], key=lambda c: (-c[2], c[1]))
        if not candidates:
            rows.append(_registry_row(prow, None, "unmatched", "no_supported_match"))
            continue

        strong = [c for c in candidates if c[2] >= STRONG_AGREEMENT]
        if len(strong) == 1:
            ci, dist, score = strong[0]
            matched_compute_ids.add(ci)
            rows.append(
                _registry_row(prow, compute.loc[ci], "high_confidence_match", None, dist, score)
            )
        elif len(strong) > 1:
            ci, dist, score = strong[0]
            matched_compute_ids.add(ci)  # claimed by this pending review, not a standalone site
            rows.append(
                _registry_row(
                    prow, compute.loc[ci], "manual_review", "multiple_plausible_nearby", dist, score
                )
            )
        else:
            ci, dist, score = candidates[0]
            matched_compute_ids.add(ci)  # claimed by this pending review, not a standalone site
            rows.append(
                _registry_row(
                    prow, compute.loc[ci], "manual_review", "weak_or_distant_match", dist, score
                )
            )

    for ci, crow in compute.iterrows():
        if ci not in matched_compute_ids:
            rows.append(_registry_row(None, crow, "compute_only_no_pnnl_candidate", None))

    return pd.DataFrame(rows)


def _apply_review_decisions(registry: pd.DataFrame, compute: pd.DataFrame, decisions_path: Path) -> pd.DataFrame:
    """Apply a completed match_review_queue review: 'confirm' rows become a
    high_confidence_match; 'reject' rows are split back into two
    independent sites (the proposed merge was wrong). Rows not mentioned in
    the decisions file pass through unchanged, so any still-unreviewed
    manual_review rows keep showing up in match_review_queue.csv."""
    if not decisions_path.exists():
        print(f"  No review decisions file at {decisions_path} -- all manual_review rows pass through as-is.")
        return registry

    decisions = (
        pd.read_csv(decisions_path, dtype={"site_id": str})
        .set_index("site_id")["reviewer_decision"]
        .str.strip()
        .str.lower()
    )
    compute_by_id = compute.set_index("compute_atlas_id", drop=False)

    # A compute_atlas_id can be the top candidate for more than one PNNL site
    # (e.g. two PNNL buildings on a campus Compute Atlas only tracks once).
    # If one of those PNNL sites ends up confirmed/high_confidence against
    # it, a sibling that gets rejected must NOT re-add that same compute
    # record as a standalone site -- it's already claimed.
    claimed_compute_ids = set(
        registry.loc[registry["match_status"] == "high_confidence_match", "compute_atlas_id"].dropna()
    ) | set(
        registry.loc[registry["site_id"].isin(decisions[decisions == "confirm"].index), "compute_atlas_id"].dropna()
    )

    rows = []
    confirmed = rejected = 0
    for _, row in registry.iterrows():
        decision = decisions.get(row["site_id"])

        if decision == "confirm":
            row = row.copy()
            row["match_status"] = "high_confidence_match"
            row["review_required"] = False
            row["review_reason"] = "manually_confirmed"
            rows.append(row.to_dict())
            confirmed += 1
        elif decision == "reject":
            pnnl_row = row[
                ["pnnl_id", "name", "operator", "state", "county", "latitude", "longitude",
                 "pnnl_record_count", "pnnl_geometry_types"]
            ]
            rows.append(_registry_row(pnnl_row, None, "unmatched", "manually_rejected"))
            if row["compute_atlas_id"] not in claimed_compute_ids:
                rows.append(
                    _registry_row(
                        None, compute_by_id.loc[row["compute_atlas_id"]],
                        "compute_only_no_pnnl_candidate", "manually_rejected",
                    )
                )
            rejected += 1
        else:
            rows.append(row.to_dict())

    print(f"  Applied review decisions: {confirmed} confirmed, {rejected} rejected (split into {rejected * 2} sites)")
    return pd.DataFrame(rows)


def _join_census_tracts(registry: pd.DataFrame, tracts_path) -> pd.DataFrame:
    """Attach each site's 2020 census tract (GEOID) via point-in-polygon
    join. Sites whose coordinates don't fall inside any tract (bad/missing
    coordinates, or a location outside the tract file's coverage) are left
    with a null tract_geoid rather than dropped."""
    import geopandas as gpd

    tracts = gpd.read_parquet(tracts_path)

    sites = gpd.GeoDataFrame(
        registry,
        geometry=gpd.points_from_xy(registry["longitude"], registry["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(sites, tracts, how="left", predicate="within").drop(columns="geometry")
    joined = joined[~joined.index.duplicated(keep="first")]  # a point on a shared tract edge
    joined = joined.rename(
        columns={"GEOID": "tract_geoid", "STATEFP": "tract_state_fips", "COUNTYFP": "tract_county_fips"}
    )
    return joined.drop(columns=[c for c in ("index_right",) if c in joined.columns])


def _init_earth_engine():
    load_dotenv()
    project = os.environ.get("GEE_PROJECT")
    if not project:
        raise RuntimeError(GEE_MISSING_MESSAGE)

    import ee

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(f"{GEE_MISSING_MESSAGE}\n\n(underlying error: {exc})") from exc
    return ee


def _sample_nlcd_year(ee, points_fc, year: int) -> dict[str, int]:
    """Sample one year's NLCD image at every point in points_fc. getInfo()
    is a blocking network call with no built-in timeout, so a stalled
    connection can hang forever — wrap it with a timeout and a few retries
    so a bad connection fails loudly instead of hanging silently."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    image = (
        ee.ImageCollection(NLCD_COLLECTION)
        .filter(ee.Filter.eq("year", year))
        .first()
        .select([0], ["lc"])
    )
    sampled = image.sampleRegions(collection=points_fc, scale=NLCD_SAMPLE_SCALE_M, geometries=False)

    for attempt in range(1, NLCD_MAX_RETRIES + 1):
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(sampled.getInfo)
        try:
            info = future.result(timeout=NLCD_CALL_TIMEOUT_S)
            pool.shutdown(wait=False)
            return {f["properties"]["site_id"]: f["properties"]["lc"] for f in info["features"]}
        except FutureTimeoutError:
            pool.shutdown(wait=False)
            print(f"    year {year}: timed out after {NLCD_CALL_TIMEOUT_S}s (attempt {attempt}/{NLCD_MAX_RETRIES})")

    raise RuntimeError(f"NLCD sampleRegions for year {year} timed out {NLCD_MAX_RETRIES}x in a row")


def _infer_construction_year(series: list[tuple[int, int | None]]) -> tuple[int | None, str]:
    """series: [(year, nlcd_class_or_None), ...] sorted ascending by year."""
    if series[0][1] in NLCD_DEVELOPED_CLASSES:
        return series[0][0], "built_before_window"  # already developed at window start

    for i, (year, cls) in enumerate(series):
        if cls in NLCD_DEVELOPED_CLASSES:
            following = [c for _, c in series[i : i + 2]]
            confidence = "high" if all(c in NLCD_DEVELOPED_CLASSES for c in following) else "low"
            return year, confidence

    return None, "no_transition_detected"


def _estimate_construction_years(registry: pd.DataFrame, min_year: int, max_year: int) -> pd.DataFrame:
    ee = _init_earth_engine()

    sites = registry.dropna(subset=["latitude", "longitude"])[["site_id", "latitude", "longitude"]]
    site_ids = sites["site_id"].tolist()

    year_to_class: dict[int, dict[str, int]] = {}
    for start in range(0, len(sites), NLCD_BATCH_SIZE):
        batch = sites.iloc[start : start + NLCD_BATCH_SIZE]
        points_fc = ee.FeatureCollection(
            [
                ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]), {"site_id": row.site_id})
                for row in batch.itertuples()
            ]
        )
        batch_start_t = time.time()
        for year in range(min_year, max_year + 1):
            year_t0 = time.time()
            year_to_class.setdefault(year, {}).update(_sample_nlcd_year(ee, points_fc, year))
            print(f"    year {year}: {time.time() - year_t0:.1f}s")
        print(
            f"  NLCD: sampled sites {start}-{start + len(batch)} of {len(sites)} "
            f"({time.time() - batch_start_t:.0f}s)"
        )

    rows = []
    for site_id in site_ids:
        series = [(y, year_to_class[y].get(site_id)) for y in range(min_year, max_year + 1)]
        construction_year, confidence = _infer_construction_year(series)
        rows.append(
            {
                "site_id": site_id,
                "construction_year": construction_year,
                "construction_year_confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def _match_epoch_ai(registry: pd.DataFrame, epoch: pd.DataFrame) -> pd.DataFrame:
    """Name/operator-only match against the registry (Epoch AI publishes no
    coordinates). For each Epoch AI entry, accept the best-scoring registry
    site only if it clears EPOCH_STRONG_AGREEMENT with a clear margin over
    the runner-up — otherwise skip it rather than guess."""
    registry_tokens = [
        _normalize_tokens(r.get("name")) | _normalize_tokens(r.get("operator"))
        for _, r in registry.iterrows()
    ]
    site_ids = registry["site_id"].tolist()

    rows = []
    for _, erow in epoch.iterrows():
        e_tokens = (
            _normalize_tokens(erow["name"])
            | _normalize_tokens(erow.get("owner"))
            | _normalize_tokens(erow.get("users"))
        )
        if not e_tokens:
            continue
        scores = [
            len(e_tokens & t) / len(e_tokens | t) if t else 0.0 for t in registry_tokens
        ]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_score = scores[ranked[0]]
        runner_up = scores[ranked[1]] if len(ranked) > 1 else 0.0
        if top_score >= EPOCH_STRONG_AGREEMENT and (top_score - runner_up) >= EPOCH_AMBIGUITY_MARGIN:
            rows.append(
                {
                    "site_id": site_ids[ranked[0]],
                    "epoch_match_name": erow["name"],
                    "epoch_construction_year": int(erow["epoch_operational_year"]),
                    "epoch_match_score": top_score,
                }
            )
    return pd.DataFrame(rows)


def _apply_epoch_ai_overrides(registry: pd.DataFrame, epoch_matches: pd.DataFrame) -> pd.DataFrame:
    registry = registry.copy()
    registry["construction_year_source"] = "nlcd"
    if epoch_matches.empty:
        registry["epoch_match_name"] = None
        return registry

    registry = registry.merge(
        epoch_matches[["site_id", "epoch_construction_year", "epoch_match_name", "epoch_match_score"]],
        on="site_id",
        how="left",
    )
    has_match = registry["epoch_construction_year"].notna()
    registry.loc[has_match, "construction_year"] = registry.loc[has_match, "epoch_construction_year"]
    registry.loc[has_match, "construction_year_confidence"] = "high"
    registry.loc[has_match, "construction_year_source"] = "epoch_ai"
    return registry.drop(columns=["epoch_construction_year"])


def main() -> None:
    cfg = load_config()
    interim_dir = ROOT / cfg["paths"]["interim"]
    processed_dir = ROOT / cfg["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    pnnl = pd.read_parquet(interim_dir / "pnnl_sites.parquet")
    compute = pd.read_parquet(interim_dir / "compute_atlas_data_centers.parquet")

    max_distance_m = cfg["registry"]["dedup_distance_m"]
    registry = _build_registry(pnnl, compute, max_distance_m)

    print("Applying manual match-review decisions (if any)...")
    decisions_path = ROOT / "data" / "manual" / "registry_review_decisions.csv"
    registry = _apply_review_decisions(registry, compute, decisions_path)

    registry = _join_census_tracts(registry, interim_dir / "census_tracts.parquet")
    unmatched_tracts = registry["tract_geoid"].isna().sum()
    print(f"{unmatched_tracts} / {len(registry)} sites did not fall inside any census tract")

    print("Estimating construction years from Annual NLCD...")
    years = cfg["years"]
    construction = _estimate_construction_years(registry, years["nlcd_min_year"], years["max_construction"])
    registry = registry.merge(construction, on="site_id", how="left")
    print(registry["construction_year_confidence"].value_counts().to_string())

    print("Cross-checking against Epoch AI data centers...")
    epoch = pd.read_parquet(interim_dir / "epoch_ai_construction_dates.parquet")
    epoch_matches = _match_epoch_ai(registry, epoch)
    registry = _apply_epoch_ai_overrides(registry, epoch_matches)
    print(f"  {len(epoch_matches)} / {len(epoch)} Epoch AI sites matched to a registry site")
    print(registry["construction_year_source"].value_counts().to_string())

    registry_parquet = processed_dir / "dc_registry.parquet"
    registry_csv = processed_dir / "dc_registry.csv"
    registry.to_parquet(registry_parquet, index=False)
    registry.to_csv(registry_csv, index=False)

    review_queue = registry[registry["review_required"]].sort_values(
        ["match_status", "match_distance_m"]
    )
    review_csv = processed_dir / "match_review_queue.csv"
    review_queue.to_csv(review_csv, index=False)

    print(f"{len(registry)} sites -> {registry_csv}")
    print(registry["match_status"].value_counts().to_string())
    print(f"{len(review_queue)} rows need manual review -> {review_csv}")


if __name__ == "__main__":
    main()
