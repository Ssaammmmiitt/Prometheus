//defining the bbox
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;

//create rectangle
var rectangle = ee.Geometry.Rectangle([minLong,minLat,maxLong,maxLat]);


//adding rectangle to map
Map.centerObject(rectangle);

// Load SRTM DEM Dataset
// Dataset ID: USGS/SRTMGL1_003
// This is a single image (not a collection)


var dem = ee.Image('USGS/SRTMGL1_003');



// Clip DEM to Nepal bounding box
var demClipped = dem.clip(rectangle).rename('elevation_m');


// Visualize DEM (meters)
var demVis = {
  min: 0,
  max: 6000,
  palette: ['ffffff', 'cfcfcf', '9e9e9e', '6f6f6f', '3f3f3f', '000000']
};

Map.addLayer(demClipped, demVis, 'Elevation (m) Nepal');

//export
Export.image.toDrive({
  image: demClipped,
  description: 'Nepal_Elevation_SRTM_1km',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Nepal_Elevation_SRTM_1km',
  region: rectangle,
  scale: 1000,     // 1 km resolution to align with other layers
  crs: 'EPSG:4326',
  maxPixels: 1e13
});