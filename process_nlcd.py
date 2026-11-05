"""
Convert Earth Engine NLCD tract histograms into a clean tract-year land-cover table.

Input : CSVs exported from gee_nlcd_export_FINAL.js, in ./nlcd_tract_exports/
        Columns: GEOID, landcover  (landcover = pixel counts per class, e.g. {23=1232.5, 24=1153.6})
        File names: nlcd_YYYY_stateFF.csv
Output: nlcd_tract_landcover_clean.csv  (one row per tract per NLCD year)

Run from the folder that contains process_nlcd.py:
    python3 process_nlcd.py
"""
import glob
import json
import os
import re
import numpy as np
import pandas as pd

IN_DIR = 'nlcd_tract_exports'
OUT = 'nlcd_tract_landcover_clean.csv'

GROUPS = {
    'pct_developed': [21, 22, 23, 24],
    'pct_forest': [41, 42, 43],
    'pct_shrub_grass': [52, 71],
    'pct_agriculture': [81, 82],
    'pct_wetland': [90, 95],
    'pct_water': [11],
    'pct_barren': [31],
}


def parse_hist(s):
    """Read a histogram in either Earth Engine format.
    JSON style:  {"21": 512, "41": 88}
    Java style:  {21=512.0, 41=88.0}   <- what the CSV export produces
    Returns a dict {class_code: pixel_count}, an empty dict if blank, or None if unreadable."""
    if pd.isna(s) or str(s).strip() in ('', '{}'):
        return {}
    text = str(s).strip()
    try:
        return {int(float(k)): float(v) for k, v in json.loads(text).items()}
    except Exception:
        pass
    found = re.findall(r'(\d+)\s*=\s*([0-9.eE+-]+)', text)
    if not found:
        return None
    return {int(k): float(v) for k, v in found}


rows = []
problems = []
for f in sorted(glob.glob(os.path.join(IN_DIR, '*.csv'))):
    m = re.search(r'nlcd_(\d{4})', os.path.basename(f))
    if not m:
        problems.append((os.path.basename(f), 'file name has no year'))
        continue
    year = int(m.group(1))
    df = pd.read_csv(f, dtype={'GEOID': str}, low_memory=False)
    for _, r in df.iterrows():
        h = parse_hist(r.get('landcover'))
        if h is None:
            problems.append((os.path.basename(f), r.get('GEOID')))
            continue
        total = sum(h.values())
        out = {'GEOID': str(r['GEOID']).zfill(11), 'nlcd_year': year, 'total_pixels': round(total, 1)}
        for g, codes in GROUPS.items():
            out[g] = round(100 * sum(h.get(c, 0) for c in codes) / total, 2) if total > 0 else np.nan
        rows.append(out)

res = pd.DataFrame(rows)
res = res.drop_duplicates(['GEOID', 'nlcd_year'])
res['flag_low_pixels'] = res.total_pixels < 100
res.to_csv(OUT, index=False)

print('rows:', len(res), '| tracts:', res.GEOID.nunique(), '| years:', sorted(res.nlcd_year.unique()))
print('GEOID lengths:', res.GEOID.str.len().value_counts().to_dict())
print('duplicate GEOID-year rows:', int(res.duplicated(['GEOID', 'nlcd_year']).sum()))
print('total_pixels median:', res.total_pixels.median(), '| min:', res.total_pixels.min(), '| max:', res.total_pixels.max())
print('low-pixel tracts flagged:', int(res.flag_low_pixels.sum()))
print('unreadable rows:', len(problems))
if problems:
    print('first problems:', problems[:5])
