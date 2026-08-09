/**
 * Prometheus — MODIS LST 8-day (MOD11A2) at native 1 km.
 *
 * Drive folder: Prometheus_GEE/lst
 * Bands: LST_Day_C, LST_Night_C (°C)
 *
 * Same fix as ndvi: never clip sinusoidals with a WGS84 rectangle;
 * use Export.region only.
 */

var minLon = 80.0129;
var minLat = 26.3386;
var maxLon = 88.2056;
var maxLat = 30.5158;
var roi = ee.Geometry.BBox(minLon, minLat, maxLon, maxLat);

// Match configs/base.yaml. For 2026-only backfill: START_YEAR = END_YEAR = 2026.
var START_YEAR = 2016;
var END_YEAR = 2026;
var DRIVE_FOLDER = 'Prometheus_GEE/lst';
var SCALE = 1000;
var CRS = 'EPSG:4326';

function kelvinToC(img) {
  var day = img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('LST_Day_C');
  var night = img.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('LST_Night_C');
  return ee.Image.cat([day, night])
    .copyProperties(img, ['system:time_start', 'system:index']);
}

var start = ee.Date.fromYMD(START_YEAR, 1, 1);
var end = ee.Date.fromYMD(END_YEAR, 5, 31).advance(1, 'day');

var col = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterDate(start, end)
  .filterBounds(roi)
  .filter(ee.Filter.calendarRange(1, 5, 'month'))
  .map(kelvinToC)
  .sort('system:time_start');

var times = col.aggregate_array('system:time_start').getInfo();
var n = times.length;
var list = col.toList(n);

print('LST images to export:', n);

function yyyymmdd(ms) {
  var d = new Date(ms);
  var y = d.getUTCFullYear();
  var m = d.getUTCMonth() + 1;
  var day = d.getUTCDate();
  var mm = m < 10 ? '0' + m : '' + m;
  var dd = day < 10 ? '0' + day : '' + day;
  return '' + y + mm + dd;
}

for (var i = 0; i < n; i++) {
  var img = ee.Image(list.get(i));
  var dateStr = yyyymmdd(times[i]);
  Export.image.toDrive({
    image: img.toFloat(),
    description: 'lst_' + dateStr,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'lst_' + dateStr,
    region: roi,
    scale: SCALE,
    crs: CRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });
}

print('Queued LST exports → Drive/' + DRIVE_FOLDER + ' scale=' + SCALE);
