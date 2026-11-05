# NLCD land cover by census tract

Land-cover percentages per census tract for the lower 48 states and DC, from the NLCD 2019 release, for 2008, 2011, 2013, 2016, and 2019.

## Files
- gee_nlcd_export_FINAL.js: Google Earth Engine script. Run it in the Earth Engine code editor. It exports one CSV per state and year to Google Drive (94 tasks).
- process_nlcd.py: Cleans the exports into one table with one row per tract per year. Requires Python 3 and pandas.
- nlcd_export_plan.csv: Which state and year exports were run, and why.

## How to run
1. Run gee_nlcd_export_FINAL.js in the Earth Engine code editor. Keep TEST_MODE = true to check one task first, then set it to false to run all 94.
2. Download the CSVs from Google Drive into a folder named nlcd_tract_exports.
3. In that folder's parent directory, run: python3 process_nlcd.py

## Notes
- Alaska and Hawaii have no NLCD data in this release.
- Pixel totals reflect full tract area. Use the percentage columns for comparisons.
- Tracts with fewer than 100 pixels are flagged (flag_low_pixels).
- The cleaned output (nlcd_tract_landcover_clean.csv) is shared separately because of its size.
