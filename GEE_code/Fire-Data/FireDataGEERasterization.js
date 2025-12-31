var roi = ee.Geometry.Rectangle([80.018, 26.347, 88.201, 30.447]);
Map.centerObject(roi, 6);
Map.addLayer(roi, {color: 'blue'}, 'ROI');

var nepal = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Nepal'))
  .geometry();
Map.addLayer(nepal, {color: 'green'}, 'Nepal boundary');

// Nepal mask image (1 inside Nepal, masked outside)
var nepalMask = ee.Image.constant(1).clip(nepal).mask();

var fires = ee.FeatureCollection('projects/wildfire-478417/assets/fire_2018_nepal_clean_conf50');

function fireRasterForMonth(year, month) {
  var mm = (month < 10 ? '0' + month : '' + month);
  var ym = year + '-' + mm;

  var firesMonth = fires.filter(ee.Filter.stringStartsWith('acq_date', ym));
  print('Fires in ' + ym, firesMonth.size());

  // Create binary raster on ROI grid, then mask to Nepal
  var fireImg = ee.Image(0).byte()
    .paint(firesMonth, 1)
    .rename('fire')
    .clip(roi) // keep ROI extent
    .updateMask(nepalMask) // keep only Nepal pixels
    .reproject({crs: 'EPSG:4326', scale: 1000});

  return fireImg;
}

// Visual check
Map.addLayer(
  fireRasterForMonth(2018, 4),
  {min: 0, max: 1, palette: ['white', 'red']},
  'Fire April 2018'
);

// Export Mar, Apr, May using ROI region (IMPORTANT)
[3, 4, 5].forEach(function(m) {
  var mm = (m < 10 ? '0' + m : '' + m);

  Export.image.toDrive({
    image: fireRasterForMonth(2018, m),
    description: 'fire_2018_' + mm + '_roiAligned',
    folder: 'GEE_Exports',
    fileNamePrefix: 'fire_2018_' + mm + '_roiAligned',
    region: roi, // export the same bbox as other layers
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});
