// NLCD land cover by census tract: minimum plan (94 tasks)
// Output columns: GEOID, landcover (histogram of pixel counts per class)

var TRACTS_ASSET = 'TIGER/2020/TRACT';
var NLCD_COLLECTION = 'USGS/NLCD_RELEASES/2019_REL/NLCD';

// Which states to export for each NLCD year
var PLAN = {
  2008: ['05', '13', '16', '19', '30', '39', '40', '46', '48', '49', '51', '56'],
  2011: ['04', '08', '18', '24', '32', '36', '41', '45', '51', '53'],
  2013: ['12', '19', '31', '32', '36', '39', '40', '41', '45', '47', '48', '49', '51', '53', '56'],
  2016: ['13', '32', '39', '41', '48', '51', '53'],
  2019: ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19',
         '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
         '36', '37', '38', '39', '40', '41', '42', '45', '46', '47', '48', '49', '50', '51', '53', '54',
         '55', '56']
};

// true = export ONE task (Arkansas 2008) to check the output first
// false = export all 94 tasks
var TEST_MODE = true;

var tracts = ee.FeatureCollection(TRACTS_ASSET);
var nlcd = ee.ImageCollection(NLCD_COLLECTION);

// Build the list of [year, state] tasks
var tasks = [];
Object.keys(PLAN).forEach(function(y) {
  PLAN[y].forEach(function(st) {
    tasks.push([parseInt(y, 10), st]);
  });
});
var runList = TEST_MODE ? tasks.slice(0, 1) : tasks;

runList.forEach(function(t) {
  var year = t[0];
  var st = t[1];

  var lc = nlcd
    .filter(ee.Filter.calendarRange(year, year, 'year'))
    .first()
    .select('landcover');

  var hist = lc.reduceRegions({
    collection: tracts.filter(ee.Filter.eq('STATEFP', st)),
    reducer: ee.Reducer.frequencyHistogram(),
    scale: 30,
    tileScale: 4
  });

  Export.table.toDrive({
    collection: hist.select(['GEOID', 'histogram'], ['GEOID', 'landcover'], false),
    description: 'nlcd_' + year + '_state' + st,
    folder: 'nlcd_tract_exports',
    fileFormat: 'CSV'
  });
});

print('Tasks created:', runList.length);
