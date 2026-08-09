/**
 * Prometheus — Static terrain + landcover (one-time export).
 *
 * Drive: Prometheus_GEE/static
 *   elev_slope_aspect.tif  — elev (m), slope (deg), aspect (deg)
 *   twi.tif                — topographic wetness index (approx)
 *   worldcover_frac.tif    — 11 landcover fraction bands at 1 km
 *
 * scale: 1000 (project 1 km grid)
 */

var minLon = 80.0129, minLat = 26.3386, maxLon = 88.2056, maxLat = 30.5158;
var roi = ee.Geometry.Rectangle([minLon, minLat, maxLon, maxLat], 'EPSG:4326', false);
var DRIVE_FOLDER = 'Prometheus_GEE/static';
var SCALE = 1000;
var CRS = 'EPSG:4326';

// ---- SRTM elev / slope / aspect ----
var elev = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elev');
var terrain = ee.Terrain.products(elev);
var slope = terrain.select('slope').rename('slope');
var aspect = terrain.select('aspect').rename('aspect');
var esa = elev.addBands([slope, aspect]).clip(roi).toFloat();

Export.image.toDrive({
  image: esa,
  description: 'static_elev_slope_aspect',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'elev_slope_aspect',
  region: roi,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// ---- Approx TWI: ln(a / tan(beta)); a from flow accumulation (HydroSHEDS-style via ee)
// Simple proxy: use slope + small constant for runoff (good enough for tree models)
var slopeRad = slope.multiply(Math.PI / 180);
var tanb = slopeRad.tan().max(0.001);
// Accumulated "upslope" proxy via focal mean elevation position (not full flow routing)
var flowProxy = elev.subtract(elev.focalMin(5, 'square', 'pixels')).rename('flow_proxy').max(1);
var twi = flowProxy.divide(tanb).log().rename('twi').clip(roi).toFloat();

Export.image.toDrive({
  image: twi,
  description: 'static_twi',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'twi',
  region: roi,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// ---- ESA WorldCover 10 m → 1 km class fractions ----
// Map: https://esa-worldcover.org (classes 10,20,...,100,95)
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').clip(roi);
var classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100];
var names = [
  'tree', 'shrub', 'grass', 'crop', 'built', 'bare',
  'snow', 'water', 'wetland', 'mangrove', 'moss'
];

var fracs = ee.Image.cat(classes.map(function (c, idx) {
  var binary = wc.eq(c); // 10 m
  // Fraction of class in each 1 km pixel via reduceResolution
  return binary
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 65536})
    .reproject({crs: CRS, scale: SCALE})
    .rename(names[idx]);
})).clip(roi).toFloat();

Export.image.toDrive({
  image: fracs,
  description: 'static_worldcover_frac',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'worldcover_frac',
  region: roi,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

print('Queued 3 static exports → Drive/' + DRIVE_FOLDER);
Map.centerObject(roi, 7);
Map.addLayer(esa.select('elev'), {min: 0, max: 5000}, 'elev');
