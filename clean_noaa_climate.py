"""
Clean NOAA nClimDiv county climate files into an annual table and 1991-2020 normals.

Input : the four county files from NOAA's climdiv directory, in the same folder:
          climdiv-tmpccy-*   (average temperature)
          climdiv-tmaxcy-*   (maximum temperature)
          climdiv-tmincy-*   (minimum temperature)
          climdiv-pcpncy-*   (precipitation)
Output: noaa_county_annual_1975_2025.csv   one row per county per year
        noaa_county_normals_1991_2020.csv  30-year county normals

Run:    python3 clean_noaa_climate.py
"""
import glob
import os
import numpy as np
import pandas as pd

IN_DIR = '.'
YEAR_MIN, YEAR_MAX = 1975, 2025
MISSING = (-99.9, -9.99)

# NOAA's state codes are alphabetical (01=AL ... 48=WY, 49=HI, 50=AK). Map to FIPS.
ABBR = ["AL","AZ","AR","CA","CO","CT","DE","FL","GA","ID","IL","IN","IA","KS","KY","LA",
        "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND",
        "OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","HI","AK"]
FIPS = ["01","04","05","06","08","09","10","12","13","16","17","18","19","20","21","22",
        "23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38",
        "39","40","41","42","44","45","46","47","48","49","50","51","53","54","55","56","15","02"]
STATE_MAP = {f"{i+1:02d}": (ABBR[i], FIPS[i]) for i in range(50)}


def find(tag):
    matches = glob.glob(os.path.join(IN_DIR, f'climdiv-{tag}-*'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one file for {tag}, found {len(matches)}')
    return matches[0]


def read_variable(path, var):
    """Each line: 2-digit state, 3-digit county, 2-digit element, 4-digit year, then 12 monthly values (7 characters each)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            state, county, year = line[0:2], line[2:5], int(line[7:11])
            vals = [float(line[11 + 7 * i: 18 + 7 * i]) for i in range(12)]
            vals = [np.nan if v in MISSING else v for v in vals]
            rows.append([state, county, year, *vals])
    cols = ['nclimdiv_state', 'county_code', 'year'] + [f'{var}_m{m:02d}' for m in range(1, 13)]
    return pd.DataFrame(rows, columns=cols)


def annual(df, var, how):
    block = df[[f'{var}_m{m:02d}' for m in range(1, 13)]]
    complete = block.notna().all(axis=1)  # a year counts only if all 12 months are present
    agg = block.mean(axis=1) if how == 'mean' else block.sum(axis=1)
    return agg.where(complete)


keys = ['nclimdiv_state', 'county_code', 'year']
df = (read_variable(find('tmpccy'), 'tavg')
      .merge(read_variable(find('tmaxcy'), 'tmax'), on=keys)
      .merge(read_variable(find('tmincy'), 'tmin'), on=keys)
      .merge(read_variable(find('pcpncy'), 'pcp'), on=keys))

df['tavg_annual_F'] = annual(df, 'tavg', 'mean')
df['tmax_annual_F'] = annual(df, 'tmax', 'mean')
df['tmin_annual_F'] = annual(df, 'tmin', 'mean')
df['pcp_annual_in'] = annual(df, 'pcp', 'sum')

df['state_abbr'] = df.nclimdiv_state.map(lambda s: STATE_MAP[s][0])
df['state_fips'] = df.nclimdiv_state.map(lambda s: STATE_MAP[s][1])
# County codes match FIPS for every state, including Alaska's NWS borough/census-area codes.
df['county_fips5'] = df.state_fips + df.county_code

out = (df[(df.year >= YEAR_MIN) & (df.year <= YEAR_MAX)]
       [['state_abbr', 'state_fips', 'county_fips5', 'nclimdiv_state', 'county_code', 'year',
         'tavg_annual_F', 'tmax_annual_F', 'tmin_annual_F', 'pcp_annual_in']]
       .sort_values(['county_fips5', 'year']))
out.to_csv('noaa_county_annual_1975_2025.csv', index=False)

# 1991-2020 normals: mean of annual values, keep counties with at least 25 of 30 years
n = out[(out.year >= 1991) & (out.year <= 2020)]
norm = (n.groupby(['state_abbr', 'state_fips', 'county_fips5'])
          .agg(tavg_norm_F=('tavg_annual_F', 'mean'), tmax_norm_F=('tmax_annual_F', 'mean'),
               tmin_norm_F=('tmin_annual_F', 'mean'), pcp_norm_in=('pcp_annual_in', 'mean'),
               years_used=('tavg_annual_F', 'count'))
          .reset_index())
norm = norm[norm.years_used >= 25].drop(columns='years_used').round(2)
norm.to_csv('noaa_county_normals_1991_2020.csv', index=False)

print('annual rows:', len(out), '| counties:', out.county_fips5.nunique())
print('duplicate county-year keys:', int(out.duplicated(['county_fips5', 'year']).sum()))
print('normals:', len(norm))
print('missing tavg share:', round(out.tavg_annual_F.isna().mean(), 4))
