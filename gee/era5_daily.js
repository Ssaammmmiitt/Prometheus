/**
 * Prometheus — ERA5-Land DAILY at NATIVE ~9 km (scale 11132).
 *
 * Run in https://code.earthengine.google.com
 * Then: Tasks panel → Start (or Run all).
 *
 * CRITICAL: scale = 11132 (~0.1°). Do NOT use 1000.
 *
 * Output layout (Google Drive folder Prometheus_GEE/era5):
 *   era5_2016_01.tif  — multi-band GeoTIFF for Jan 2016
 *   bands named like: 20160101_t2m_max, 20160101_t2m_min, ...
 *   (one set of 9 vars per day in the month)
 */

var minLon = 80.0129, minLat = 26.3386, maxLon = 88.2056, maxLat = 30.5158;
var roi = ee.Geometry.Rectangle([minLon, minLat, maxLon, maxLat], 'EPSG:4326', false);

// Match configs/base.yaml years.train_*. To *add only 2026* after 2016–2025
// already exported, set both to 2026 (avoids re-queuing ~50 tasks).
var START_YEAR = 2016;
var END_YEAR = 2026;
var MONTHS = [1, 2, 3, 4, 5];
var DRIVE_FOLDER = 'Prometheus_GEE/era5';
var SCALE = 11132;
var CRS = 'EPSG:4326';

var SRC = 'ECMWF/ERA5_LAND/DAILY_AGGR';

function prep(img) {
  return ee.Image.cat([
    img.select('temperature_2m_max').subtract(273.15).rename('t2m_max'),
    img.select('temperature_2m_min').subtract(273.15).rename('t2m_min'),
    img.select('temperature_2m').subtract(273.15).rename('t2m'),
    img.select('dewpoint_temperature_2m').subtract(273.15).rename('d2m'),
    img.select('total_precipitation_sum').multiply(1000).rename('precip'), // mm
    img.select('u_component_of_wind_10m').rename('u10'),
    img.select('v_component_of_wind_10m').rename('v10'),
    img.select('volumetric_soil_water_layer_1').rename('soil_water_l1'),
    img.select('surface_pressure').rename('surface_pressure')
  ]).copyProperties(img, ['system:time_start']);
}

function pad2(n) {
  return (n < 10 ? '0' : '') + n;
}

function exportMonth(year, month) {
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');

  var days = ee.ImageCollection(SRC)
    .filterDate(start, end)
    .filterBounds(roi)
    .map(prep)
    .map(function (img) {
      // Prefix every band with YYYYMMDD so toBands() is unique & parseable
      var d = ee.Date(img.get('system:time_start'));
      var prefix = d.format('YYYYMMdd').cat('_');
      var names = img.bandNames().map(function (b) {
        return prefix.cat(ee.String(b));
      });
      return img.rename(names);
    });

  // One multi-band image for the whole month
  var stacked = days.toBands();
  // toBands keeps prior names if unique; strip leading "N_" if present
  // Actually toBands uses system:index prefix — clean that by renaming from properties.
  // Safer: iterate and addBands without system index noise.
  var list = days.toList(days.size());
  var n = days.size();
  var stackedClean = ee.Image(
    ee.List.sequence(0, n.subtract(1)).iterate(function (i, acc) {
      i = ee.Number(i);
      acc = ee.Image(acc);
      var im = ee.Image(list.get(i));
      return ee.Image(ee.Algorithms.If(i.eq(0), im, acc.addBands(im)));
    }, ee.Image.constant(0))
  );
  // Drop the dummy constant if sequence used wrong path — rebuild simply:
  stackedClean = ee.ImageCollection(list).toBands();
  // Rename: system index often is DATE — band becomes "20160101_t2m_max" already if rename worked.
  // If GEE prefixes with 0_,1_... we live with parse later in Python by split.

  var tag = year + '_' + pad2(month);
  Export.image.toDrive({
    image: stackedClean.clip(roi).toFloat(),
    description: 'era5_' + tag,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'era5_' + tag,
    region: roi,
    scale: SCALE,
    crs: CRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

for (var y = START_YEAR; y <= END_YEAR; y++) {
  for (var m = 0; m < MONTHS.length; m++) {
    exportMonth(y, MONTHS[m]);
  }
}

print('Queued', (END_YEAR - START_YEAR + 1) * MONTHS.length, 'ERA5 monthly exports → Drive/' + DRIVE_FOLDER);
print('Open Tasks → Start each (or select all → Run). Native scale:', SCALE, 'm');
