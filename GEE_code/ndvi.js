//defining the bbox
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;

//create rectangle
var rectangle = ee.Geometry.Rectangle([minLong,minLat,maxLong,maxLat]);


//adding rectangle to map
Map.centerObject(rectangle);


var ndvi2018 = ee.ImageCollection('MODIS/061/MOD13Q1')
                  .filter(ee.Filter.date('2018-03-01', '2018-05-31'));
var ndviBand = ndvi2018.select('NDVI');
var ndviClipped = ndviBand.map(function(image) {
  return image.clip(rectangle);
});

//Reduce to Single Image (Mean NDVI)
// Average NDVI across March–May 2018
var ndviMean = ndviClipped.mean();



//Visualization Settings
var ndviVis = {
  min: 0,
  max: 8000, // MODIS NDVI is scaled by 10,000
  palette: [
    'ffffff', 'ce7e45', 'df923d', 'f1b555', 'fcd163',
    '99b718', '74a901', '66a000', '529400', '3e8601',
    '207401', '056201', '004c00'
  ]
};


// Display NDVI on Map
Map.addLayer(ndviMean, ndviVis, 'Mean NDVI (Mar–May 2018)');

//Export
Export.image.toDrive({
  image: ndviMean,
  description: 'Nepal_NDVI_2018_MarMay',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Nepal_NDVI_2018_MarMay',
  region: rectangle,
  scale: 1000, // 1 km resolution
  crs: 'EPSG:4326',
  maxPixels: 1e13
});


