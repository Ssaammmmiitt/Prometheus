// -----------------------------------------
// Nepal ROI
// -----------------------------------------
var minLat = 26.347;
var minLong = 80.018;
var maxLat = 30.447;
var maxLong = 88.201;

var roi = ee.Geometry.Rectangle([minLong, minLat, maxLong, maxLat]);
Map.centerObject(roi, 6);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// -----------------------------------------
// CHANGE YEAR HERE
// -----------------------------------------
var year = 2020;

// -----------------------------------------
// Date range: Jan 1 – May 31
// -----------------------------------------
var startDate = ee.Date.fromYMD(year, 1, 1);
var endDate   = ee.Date.fromYMD(year, 5, 31);

// -----------------------------------------
// ERA5-Land daily variables
// -----------------------------------------
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate(startDate, endDate)
  .filterBounds(roi)
  .select(['temperature_2m', 'dewpoint_temperature_2m']);

// -----------------------------------------
// Helper: compute VPD (kPa)
// -----------------------------------------
function computeVPD(img) {
  var T  = img.select('temperature_2m').subtract(273.15);
  var Td = img.select('dewpoint_temperature_2m').subtract(273.15);

  var es = T.expression(
    '0.6108 * exp((17.27 * T) / (T + 237.3))',
    {T: T}
  );

  var ea = Td.expression(
    '0.6108 * exp((17.27 * Td) / (Td + 237.3))',
    {Td: Td}
  );

  return es.subtract(ea)
           .rename('VPD')
           .copyProperties(img, ['system:time_start']);
}

// -----------------------------------------
// Create 16-day windows manually
// -----------------------------------------
var days = ee.List.sequence(0, endDate.difference(startDate, 'day'), 16);

days.getInfo().forEach(function(d) {

  var winStart = startDate.advance(d, 'day');
  var winEnd   = winStart.advance(16, 'day');

  var vpd16 = era5
    .filterDate(winStart, winEnd)
    .map(computeVPD)
    .mean()
    .clip(roi)
    .rename('VPD');

  var dateStr = winStart.format('YYYYMMdd').getInfo();
  var name = 'vpd16_' + year + '_' + dateStr;

  Export.image.toDrive({
    image: vpd16,
    description: name,
    folder: 'GEE_Exports_16/vpd16',
    fileNamePrefix: name,
    region: roi,
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});

// -----------------------------------------
// Quick visual check (SAFE version)
// -----------------------------------------
var img = ee.Image(era5.first());

var T  = img.select('temperature_2m').subtract(273.15);
var Td = img.select('dewpoint_temperature_2m').subtract(273.15);

var es = T.expression(
  '0.6108 * exp((17.27 * T) / (T + 237.3))',
  {T: T}
);

var ea = Td.expression(
  '0.6108 * exp((17.27 * Td) / (Td + 237.3))',
  {Td: Td}
);

var vpdPreview = es.subtract(ea).rename('VPD').clip(roi);

Map.addLayer(
  vpdPreview,
  {min: 0, max: 5, palette: ['blue', 'cyan', 'yellow', 'orange', 'red']},
  'VPD example'
);