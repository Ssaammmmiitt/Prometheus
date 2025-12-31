// -----------------------------------------
// Nepal bbox
// -----------------------------------------
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;

var rectangle = ee.Geometry.Rectangle([minLong, minLat, maxLong, maxLat]);
Map.centerObject(rectangle, 6);
Map.addLayer(rectangle, {color: 'red'}, 'Nepal BBOX');

// -----------------------------------------
// Helper function: monthly mean temperature (Celsius)
// -----------------------------------------
function monthlyTempC(year, month) {
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');

  var tempMeanC = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterDate(start, end)
    .select('temperature_2m')
    .map(function(img) { return img.clip(rectangle); })
    .mean()
    .subtract(273.15)
    .rename('temp_C');

  return tempMeanC;
}

// -----------------------------------------
// Export months needed for ConvLSTM inputs
// For target months Mar-Apr-May, inputs need Jan-Feb-Mar-Apr
// -----------------------------------------
var year = 2018;
var months = [1, 2, 3, 4];

months.forEach(function(m) {
  var img = monthlyTempC(year, m);

  var mm = (m < 10 ? '0' + m : '' + m);

  Export.image.toDrive({
    image: img,
    description: 'tempC_' + year + '_' + mm,
    folder: 'GEE_Exports',
    fileNamePrefix: 'tempC_' + year + '_' + mm,
    region: rectangle,
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});

// Optional: visualize one month to confirm it looks right
var tempVis = {
  min: -10,
  max: 30,
  palette: ['0b1d51', '1565c0', '42a5f5', 'a5d6a7', 'fff59d', 'ffb74d', 'e53935']
};
Map.addLayer(monthlyTempC(2018, 3), tempVis, 'TempC 2018-03');
