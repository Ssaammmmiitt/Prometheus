var roi = ee.Geometry.Rectangle([80.018, 26.347, 88.201, 30.447]);

var nepal = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Nepal'));

var mask01 = ee.Image(0).byte()
  .paint(nepal, 1)
  .clip(roi)
  .rename('nepal_mask');

Map.centerObject(roi, 6);
Map.addLayer(mask01, {min: 0, max: 1}, 'Nepal mask 0 1');

Export.image.toDrive({
  image: mask01,
  description: 'nepal_mask_1km_roiAligned',
  folder: 'GEE_Exports',
  fileNamePrefix: 'nepal_mask_1km_roiAligned',
  region: roi,
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
