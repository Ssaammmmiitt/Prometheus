//defining the bbox
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;

//create rectangle
var rectangle = ee.Geometry.Rectangle([minLong,minLat,maxLong,maxLat]);


//adding rectangle to map
Map.centerObject(rectangle);


// ERA5 Land daily temperature, Mar-May 2018
var tempMeanC = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate('2018-03-01', '2018-05-31')
  .select('temperature_2m')
  .map(function(img) {
    return img.clip(rectangle);
  })
  .mean()
  .subtract(273.15)
  .rename('temp_C');

// Simple visualization with real Nepal range
var tempVis = {
  min: -10,
  max: 30,
  palette: ['0b1d51', '1565c0', '42a5f5', 'a5d6a7', 'fff59d', 'ffb74d', 'e53935']
};

Map.addLayer(tempMeanC, tempVis, 'Mean Temp (C) Mar-May 2018');

// STEP 10: Export to Google Drive as GeoTIFF

Export.image.toDrive({
  image: tempMeanC,
  description: 'Nepal_TemperatureC_2018_MarMay',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Nepal_TemperatureC_2018_MarMay',
  region: rectangle,
  scale: 1000,        // 1 km resolution (consistent with NDVI export)
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

