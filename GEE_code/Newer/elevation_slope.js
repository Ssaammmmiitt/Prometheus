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
// Elevation: SRTM (meters)
// ============================
var dem = ee.Image('USGS/SRTMGL1_003')
  .select('elevation')
  .clip(roi)
  .rename('elevation_m');

// ============================
// Slope (degrees) from DEM
// ============================
var slope = ee.Terrain.slope(dem)
  .clip(roi)
  .rename('slope_deg');

// ============================
// Export Elevation
// ============================
Export.image.toDrive({
  image: dem,
  description: 'elevation_static_srtm',
  folder: 'GEE_Exports/static',
  fileNamePrefix: 'elevation_static_srtm',
  region: roi,
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// ============================
// Export Slope
// ============================
Export.image.toDrive({
  image: slope,
  description: 'slope_static_srtm',
  folder: 'GEE_Exports/static',
  fileNamePrefix: 'slope_static_srtm',
  region: roi,
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// ============================
// Quick view
// ============================
Map.addLayer(dem, {min: 0, max: 8000}, 'Elevation (m)');
Map.addLayer(slope, {min: 0, max: 60}, 'Slope (deg)');
