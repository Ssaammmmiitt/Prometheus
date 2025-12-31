//defining the bbox
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;

//create rectangle
var rectangle = ee.Geometry.Rectangle([minLong,minLat,maxLong,maxLat]);


//adding rectangle to map
Map.centerObject(rectangle);

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
var era5_2018 = era5.filterDate('2018-03-01', '2018-05-31');

// Select Precipitation Band
// Band name: total_precipitation_sum
// Units: meters (m) per day
var precip = era5_2018.select('total_precipitation_sum');

//clip to nepal bbox
var precipClipped = precip.map(function(image) {
  return image.clip(rectangle);
});


//sum precipitation indicating rainfall
var precipTotalM = precipClipped.sum();

// STEP 7: Convert meters to millimeters
// 1 meter = 1000 millimeters
var precipTotalMM = precipTotalM.multiply(1000).rename('precip_mm');


// Visualization Settings (mm)


var precipVis = {
  min: 0,
  max: 600,
  palette: [
    'ffffff',
    'ccece6',
    '99d8c9',
    '66c2a4',
    '2ca25f',
    '006d2c'
  ]
};

// Display on Map

Map.addLayer(precipTotalMM, precipVis, 'Total Precip (mm) Mar–May 2018');

//export:
Export.image.toDrive({
  image: precipTotalMM,
  description: 'Nepal_PrecipitationMM_2018_MarMay',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Nepal_PrecipitationMM_2018_MarMay',
  region: rectangle,
  scale: 1000,     // 1 km resolution
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

