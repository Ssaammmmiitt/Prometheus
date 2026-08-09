/**
 * Prometheus — MODIS NDVI/EVI 16-day (MOD13Q1) exported at 1 km (scale 1000).
 * Native product is 250 m; export scale 1000 matches the project grid spacing.
 *
 * Drive folder: Prometheus_GEE/ndvi
 * Bands: NDVI, EVI  (physical scale, ×0.0001 applied)
 *
 * FIX: do NOT img.clip(roi) — MODIS is in sinusoidal; clipping a WGS84
 * rectangle often fails with "Image.clip: Can't transform (0.0,0.0)".
 * Export.region crops the footprint safely during reproject-to-EPSG:4326.
 */

// ---------- ROI (configs/base.yaml bbox) ----------
var minLon = 80.0129;
var minLat = 26.3386;
var maxLon = 88.2056;
var maxLat = 30.5158;
// BBox is less error-prone than Rectangle(coords, proj, geodesic)
var roi = ee.Geometry.BBox(minLon, minLat, maxLon, maxLat);

// Match configs/base.yaml. For 2026-only backfill after older years exist:
// set START_YEAR = END_YEAR = 2026.
var START_YEAR = 2016;
var END_YEAR = 2026;
var DRIVE_FOLDER = 'Prometheus_GEE/ndvi';
var SCALE = 1000;
var CRS = 'EPSG:4326';

function prep(img) {
  // Keep default projection; only band math
  var ndvi = img.select('NDVI').multiply(0.0001).rename('NDVI');
  var evi = img.select('EVI').multiply(0.0001).rename('EVI');
  return ee.Image.cat([ndvi, evi])
    .copyProperties(img, ['system:time_start', 'system:index']);
}

var start = ee.Date.fromYMD(START_YEAR, 1, 1);
var end = ee.Date.fromYMD(END_YEAR, 5, 31).advance(1, 'day');

var col = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(start, end)
  .filterBounds(roi)
  .filter(ee.Filter.calendarRange(1, 5, 'month'))
  .map(prep)
  .sort('system:time_start');

// One client fetch for times (faster + more stable than getInfo per image)
var times = col.aggregate_array('system:time_start').getInfo();
var n = times.length;
var list = col.toList(n);

print('NDVI images to export:', n);
print('Cancel any old failed ndvi_* tasks, then Start the new ones from the Tasks panel.');

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
  // No .clip(roi) — region + crs handle crop + warp
  Export.image.toDrive({
    image: img.toFloat(),
    description: 'ndvi_' + dateStr,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'ndvi_' + dateStr,
    region: roi,
    scale: SCALE,
    crs: CRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });
}

print('Queued NDVI exports → Drive/' + DRIVE_FOLDER + ' scale=' + SCALE);
