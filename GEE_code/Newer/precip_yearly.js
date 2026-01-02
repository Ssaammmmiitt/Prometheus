// ============================
// Nepal ROI
// ============================
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;

var roi = ee.Geometry.Rectangle([minLong, minLat, maxLong, maxLat]);
Map.centerObject(roi, 6);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// ============================
// CHANGE YEAR HERE
// ============================
var year = 2018;

// ============================
// Date range: Jan 1 – May 31
// ============================
var startDate = ee.Date.fromYMD(year, 1, 1);
var endDate   = ee.Date.fromYMD(year, 5, 31);

// ============================
// ERA5-Land DAILY precipitation
// Correct band: total_precipitation_sum
// Units: meters
// ============================
var precipCol = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate(startDate, endDate)
  .filterBounds(roi)
  .select('total_precipitation_sum');

// ============================
// Create 16-day precipitation sums
// ============================
var days = ee.List.sequence(0, endDate.difference(startDate, 'day'), 16);

days.getInfo().forEach(function(d) {

  var winStart = startDate.advance(d, 'day');
  var winEnd   = winStart.advance(16, 'day');

  var precip16 = precipCol
    .filterDate(winStart, winEnd)
    .sum()                 // sum over 16 days
    .multiply(1000)        // meters → millimeters
    .clip(roi)
    .rename('precip_mm');

  var dateStr = winStart.format('YYYYMMdd').getInfo();
  var name = 'precip16_' + year + '_' + dateStr;

  Export.image.toDrive({
    image: precip16,
    description: name,
    folder: 'GEE_Exports/precip',
    fileNamePrefix: name,
    region: roi,
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});

// ============================
// Quick visual check
// ============================
Map.addLayer(
  precipCol.first().multiply(1000).clip(roi),
  {min: 0, max: 50, palette: ['ffffff', 'b3e5fc', '0288d1']},
  'Precip example (mm)'
);
