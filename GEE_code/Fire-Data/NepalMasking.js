var roi = ee.Geometry.Rectangle([80.018, 26.347, 88.201, 30.447]);

var nepal = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Nepal'))
  .geometry();

// Make a mask that is 1 inside Nepal and masked outside
var nepalMask = ee.Image.constant(1)
  .clip(roi)
  .updateMask(ee.Image.constant(1).clip(nepal).mask())
  .rename('nepal_mask')
  .reproject({crs: 'EPSG:4326', scale: 1000});

Map.centerObject(roi, 6);
Map.addLayer(nepalMask, {min: 0, max: 1, palette: ['white', 'green']}, 'Nepal Mask');

Export.image.toDrive({
  image: nepalMask,
  description: 'nepal_mask_1km_roiAligned',
  folder: 'GEE_Exports',
  fileNamePrefix: 'nepal_mask_1km_roiAligned',
  region: roi,
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
