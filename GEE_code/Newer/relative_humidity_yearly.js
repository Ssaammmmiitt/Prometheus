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
// CHANGE YEAR HERE
// ============================
var year = 2018;

// ============================
// Date range: Jan 1 – May 31
// ============================
var startDate = ee.Date.fromYMD(year, 1, 1);
var endDate   = ee.Date.fromYMD(year, 5, 31);

// ============================
// ERA5-Land daily variables
// temperature_2m and dewpoint_temperature_2m are in Kelvin
// ============================
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate(startDate, endDate)
  .filterBounds(roi)
  .select(['temperature_2m', 'dewpoint_temperature_2m']);

// ============================
// Compute daily Relative Humidity (%)
// RH = 100 * exp((17.625*Td)/(243.04+Td)) / exp((17.625*T)/(243.04+T))
// Using T and Td in Celsius
// ============================
function addRH(img) {
  var T  = img.select('temperature_2m').subtract(273.15);
  var Td = img.select('dewpoint_temperature_2m').subtract(273.15);

  var esTd = Td.expression('exp((a*td)/(b+td))', {
    'a': 17.625,
    'b': 243.04,
    'td': Td
  });

  var esT = T.expression('exp((a*t)/(b+t))', {
    'a': 17.625,
    'b': 243.04,
    't': T
  });

  var rh = esTd.divide(esT).multiply(100).clamp(0, 100).rename('RH');

  return rh.copyProperties(img, ['system:time_start']);
}

var rhDaily = era5.map(addRH);

// ============================
// Create 16-day windows and export mean RH
// ============================
var days = ee.List.sequence(0, endDate.difference(startDate, 'day'), 16);

days.getInfo().forEach(function(d) {

  var winStart = startDate.advance(d, 'day');
  var winEnd   = winStart.advance(16, 'day');

  var rh16 = rhDaily
    .filterDate(winStart, winEnd)
    .mean()
    .clip(roi)
    .rename('RH');

  var dateStr = winStart.format('YYYYMMdd').getInfo();
  var name = 'rh16_' + year + '_' + dateStr;

  Export.image.toDrive({
    image: rh16,
    description: name,
    folder: 'GEE_Exports/rel_humidity',
    fileNamePrefix: name,
    region: roi,
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});

// ============================
// Quick visual check
// ============================
Map.addLayer(
  rhDaily.first().clip(roi),
  {min: 0, max: 100},
  'RH example (%)'
);
