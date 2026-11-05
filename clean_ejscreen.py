"""
Clean the EJScreen 2024 tract file: fix GEOIDs, remove territories, keep study fields.

Input : EJScreen_2024_Tract_with_AS_CNMI_GU_VI.csv  (same folder)
Output: ejscreen_tract_clean_50states_dc.csv

Run:    python3 clean_ejscreen.py
"""
import pandas as pd

IN_FILE = 'EJScreen_2024_Tract_with_AS_CNMI_GU_VI.csv'
OUT_FILE = 'ejscreen_tract_clean_50states_dc.csv'
TERRITORIES = ['AS', 'GU', 'MP', 'PR', 'VI']  # not in the study; their IDs don't match standard GEOIDs

FIELDS = {
    'population_household': ['ACSTOTPOP', 'ACSTOTHH', 'ACSTOTHU'],
    'socioeconomic': ['LOWINCPCT', 'UNEMPPCT', 'LINGISOPCT', 'LESSHSPCT',
                      'UNDER5PCT', 'OVER64PCT', 'DISABILITYPCT', 'PEOPCOLORPCT'],
    'demographic_index': ['DEMOGIDX_2', 'DEMOGIDX_5'],
    'environmental': ['PM25', 'OZONE', 'DSLPM', 'RSEI_AIR', 'PTRAF', 'PRE1960PCT',
                      'PNPL', 'PRMP', 'PTSDF', 'UST', 'PWDIS', 'NO2', 'DWATER'],
}

e = pd.read_csv(IN_FILE, dtype={'ID': str}, encoding='utf-8-sig', low_memory=False)
e['GEOID'] = e['ID'].str.strip()
e = e[~e['ST_ABBREV'].isin(TERRITORIES)].copy()

bad = e[e.GEOID.str.len() != 11]
if len(bad):
    raise ValueError(f'{len(bad)} non-11-digit GEOIDs remain; check the input file')

cols = [c for group in FIELDS.values() for c in group if c in e.columns]
out = e[['GEOID', 'ST_ABBREV', 'STATE_NAME', 'CNTY_NAME', 'REGION'] + cols].rename(columns={
    'ST_ABBREV': 'state_abbr', 'STATE_NAME': 'state_name',
    'CNTY_NAME': 'county_name', 'REGION': 'epa_region'}).copy()

out['flag_no_pop'] = out['ACSTOTPOP'].isna() | (out['ACSTOTPOP'] == 0)
out['flag_missing_socio'] = out['LOWINCPCT'].isna() | out['PEOPCOLORPCT'].isna()

assert not out.GEOID.duplicated().any(), 'duplicate GEOIDs'
out.to_csv(OUT_FILE, index=False)

print('rows written:', len(out), '| states + DC:', out.state_abbr.nunique())
print('no-population tracts flagged:', int(out.flag_no_pop.sum()))
print('tracts with missing socioeconomic fields:', int(out.flag_missing_socio.sum()))
