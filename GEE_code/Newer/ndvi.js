// Nepal bbox
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;
var roi = ee.Geometry.Rectangle([minLong, minLat, maxLong, maxLat]);

Map.centerObject(roi, 6);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// Helper: make a monthly NDVI image
function monthlyNDVI(year, month) {
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');

  // MOD13Q1 NDVI is scaled by 10000, keep it scaled for now
  var ndvi = ee.ImageCollection('MODIS/061/MOD13Q1')
    .filterDate(start, end)
    .select('NDVI')
    .mean()
    .clip(roi)
    .rename('NDVI');

  return ndvi;
}

// Months you need for 2018
var year = 2018;
var months = [1, 2, 3, 4];

months.forEach(function(m) {
  var img = monthlyNDVI(year, m);

  Export.image.toDrive({
    image: img,
    description: 'ndvi_' + year + '_' + (m < 10 ? '0' + m : m),
    folder: 'GEE_Exports',
    fileNamePrefix: 'ndvi_' + year + '_' + (m < 10 ? '0' + m : m),
    region: roi,
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});

// Optional quick view for one month
Map.addLayer(monthlyNDVI(2018, 3), {min: 0, max: 8000}, 'NDVI 2018 03');
// Nepal bbox
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;
var roi = ee.Geometry.Rectangle([minLong, minLat, maxLong, maxLat]);

Map.centerObject(roi, 6);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// Helper: make a monthly NDVI image
function monthlyNDVI(year, month) {
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');

  // MOD13Q1 NDVI is scaled by 10000, keep it scaled for now
  var ndvi = ee.ImageCollection('MODIS/061/MOD13Q1')
    .filterDate(start, end)
    .select('NDVI')
    .mean()
    .clip(roi)
    .rename('NDVI');

  return ndvi;
}

// Months you need for 2018
var year = 2018;
var months = [1, 2, 3, 4];

months.forEach(function(m) {
  var img = monthlyNDVI(year, m);

  Export.image.toDrive({
    image: img,
    description: 'ndvi_' + year + '_' + (m < 10 ? '0' + m : m),
    folder: 'GEE_Exports',
    fileNamePrefix: 'ndvi_' + year + '_' + (m < 10 ? '0' + m : m),
    region: roi,
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});

// Optional quick view for one month
Map.addLayer(monthlyNDVI(2018, 3), {min: 0, max: 8000}, 'NDVI 2018 03');
