// Nepal bbox
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;
var roi = ee.Geometry.Rectangle([minLong, minLat, maxLong, maxLat]);

Map.centerObject(roi, 6);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// Helper: monthly total precipitation in millimeters
function monthlyPrecipMM(year, month) {
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');

  // total_precipitation_sum is meters per day, sum over month then convert to mm
  var precipM = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterDate(start, end)
    .select('total_precipitation_sum')
    .sum()
    .clip(roi);

  var precipMM = precipM.multiply(1000).rename('precip_mm');
  return precipMM;
}

var year = 2018;
var months = [1, 2, 3, 4];

months.forEach(function(m) {
  var img = monthlyPrecipMM(year, m);

  Export.image.toDrive({
    image: img,
    description: 'precipMM_' + year + '_' + (m < 10 ? '0' + m : m),
    folder: 'GEE_Exports',
    fileNamePrefix: 'precipMM_' + year + '_' + (m < 10 ? '0' + m : m),
    region: roi,
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});

// Optional quick view
Map.addLayer(monthlyPrecipMM(2018, 3), {min: 0, max: 400}, 'PrecipMM 2018 03');
