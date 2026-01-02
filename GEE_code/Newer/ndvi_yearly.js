
// Nepal ROI

var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;

var roi = ee.Geometry.Rectangle([minLong, minLat, maxLong, maxLat]);

Map.centerObject(roi, 6);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// CHANGE YEAR HERE

var year = 2018;


// Date range: Jan 1 – May 31

var startDate = ee.Date.fromYMD(year, 1, 1);
var endDate   = ee.Date.fromYMD(year, 5, 31);


// MODIS NDVI (16-day composite)

var ndviCol = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(startDate, endDate)
  .filterBounds(roi)
  .select('NDVI');


// Convert to list for simple export

var ndviList = ndviCol.toList(ndviCol.size());
var count = ndviCol.size().getInfo();

print('Number of 16-day composites:', count);


// Export each 16-day image

for (var i = 0; i < count; i++) {

  var img = ee.Image(ndviList.get(i))
    .clip(roi)
    .rename('NDVI');

  var date = ee.Date(img.get('system:time_start'))
                .format('YYYYMMdd')
                .getInfo();

  var name = 'ndvi16_' + year + '_' + date;

  Export.image.toDrive({
    image: img,                 // NDVI still scaled 0–10000
    description: name,
    folder: 'GEE_Exports_16',
    fileNamePrefix: name,
    region: roi,
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}


// Quick visual check

Map.addLayer(
  ee.Image(ndviList.get(0)).clip(roi),
  {min: 0, max: 8000},
  'NDVI example'
);
