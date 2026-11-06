import pandas as pd, zipfile, json
P='/mnt/data/'
def id11(s):return s.astype('string').str.replace(r'\.0$','',regex=True).str.zfill(11)
def id5(s):return s.astype('string').str.replace(r'\.0$','',regex=True).str.zfill(5)
with zipfile.ZipFile(P+'sampling.zip') as z:
 cum=pd.read_csv(z.open('sampling/cumulative_analysis.csv'),dtype={'tract_id':'string'})
 bef=pd.read_csv(z.open('sampling/before_analysis.csv'),dtype={'tract_id':'string'})
for d in [cum,bef]:
 d['tract_id']=id11(d['tract_id']);d['county_fips5']=d['tract_id'].str[:5]
nlcd=pd.read_csv(P+'nlcd_tract_landcover_clean.csv',dtype={'GEOID':'string'});nlcd['tract_id']=id11(nlcd.GEOID)
ej=pd.read_csv(P+'ejscreen_tract_clean_50states_dc.csv',dtype={'GEOID':'string'});ej['tract_id']=id11(ej.GEOID)
ej=ej.drop(columns=['GEOID'])
annual=pd.read_csv(P+'noaa_county_annual_1975_2025.csv',dtype={'county_fips5':'string'});annual['county_fips5']=id5(annual.county_fips5)
norm=pd.read_csv(P+'noaa_county_normals_1991_2020.csv',dtype={'county_fips5':'string'});norm['county_fips5']=id5(norm.county_fips5)
print('duplicates',ej.tract_id.duplicated().sum(),nlcd.duplicated(['tract_id','nlcd_year']).sum(),annual.duplicated(['county_fips5','year']).sum(),norm.county_fips5.duplicated().sum())
assert not ej.tract_id.duplicated().any() and not nlcd.duplicated(['tract_id','nlcd_year']).any() and not annual.duplicated(['county_fips5','year']).any()
ejcols=['tract_id','PM25','OZONE','DSLPM','NO2','DWATER','RSEI_AIR','PTRAF','flag_no_pop','flag_missing_socio','DEMOGIDX_2','DEMOGIDX_5']
ej2=ej[ejcols].rename(columns={c:'ej_'+c.lower() for c in ejcols if c!='tract_id'})
# Current day: 2019 land-cover is last available, and NOAA 1991-2020 normals are baseline climate.
cumn=nlcd[nlcd.nlcd_year==2019].drop(columns=['GEOID']).rename(columns={'nlcd_year':'nlcd_source_year'})
cur=cum.merge(cumn,on='tract_id',how='left',validate='one_to_one',indicator='nlcd_join').merge(ej2,on='tract_id',how='left',validate='one_to_one',indicator='ej_join').merge(norm.drop(columns=['state_abbr','state_fips']),on='county_fips5',how='left',validate='many_to_one',indicator='noaa_join')
# Historical: find most recent NLCD <= reference year, no future land-cover leakage.
land=nlcd.drop(columns=['GEOID']).sort_values('nlcd_year')
bef0=bef.reset_index(names='row_id');pairs=bef0[['row_id','tract_id','match_year']].merge(land,on='tract_id',how='left');pairs=pairs[pairs.nlcd_year.le(pairs.match_year)];pairs=pairs.sort_values(['row_id','nlcd_year']).drop_duplicates('row_id',keep='last').drop(columns=['tract_id','match_year']).rename(columns={'nlcd_year':'nlcd_source_year'})
hist=bef0.merge(pairs,on='row_id',how='left',validate='one_to_one',indicator='nlcd_join').merge(annual[['county_fips5','year','tavg_annual_F','tmax_annual_F','tmin_annual_F','pcp_annual_in']],left_on=['county_fips5','match_year'],right_on=['county_fips5','year'],how='left',validate='many_to_one',indicator='noaa_join').drop(columns=['year'])
# Present-day EJScreen retained for context only; not appropriate as historical model predictor.
hist=hist.merge(ej2,on='tract_id',how='left',validate='many_to_one',indicator='ej_join')
for name,d,base in [('cumulative',cur,cum),('historical',hist,bef)]:
 assert len(d)==len(base),(name,len(d),len(base))
 assert d['is_case'].sum()==base['is_case'].sum()
 d.to_csv(P+name+'_modeling.csv',index=False)
 print(name,'rows',len(d),'cases',int(d.is_case.sum()),'NLCD',d.nlcd_source_year.notna().sum(),'EJ',d.ej_pm25.notna().sum(),'NOAA',d.noaa_join.eq('both').sum(),'low_pixels',d.flag_low_pixels.fillna(False).sum())
 print('NLCD years',d.nlcd_source_year.value_counts(dropna=False).sort_index().to_dict())
report={'current':{'rows':len(cur),'cases':int(cur.is_case.sum()),'nlcd_joined':int(cur.nlcd_source_year.notna().sum()),'ej_joined':int(cur.ej_pm25.notna().sum()),'noaa_joined':int(cur.noaa_join.eq('both').sum()),'nlcd_low_pixels':int(cur.flag_low_pixels.fillna(False).sum())},'historical':{'rows':len(hist),'cases':int(hist.is_case.sum()),'nlcd_joined':int(hist.nlcd_source_year.notna().sum()),'ej_joined':int(hist.ej_pm25.notna().sum()),'noaa_joined':int(hist.noaa_join.eq('both').sum()),'nlcd_low_pixels':int(hist.flag_low_pixels.fillna(False).sum())},'notes':['Current land cover is 2019, not 2024.','Historical NLCD is most recent observed year <= match_year.','Historical EJScreen is contemporaneous context only and should be excluded from preconstruction modeling.','Historical observations can repeat tract_id across years; row_id preserves individual observations.','Control reuse requires grouped/geographic split to prevent leakage.','EJScreen demographic indices overlap ACS socioeconomic features.']}
with open(P+'merge_quality_report.json','w') as f:json.dump(report,f,indent=2)
