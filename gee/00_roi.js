/**
 * Prometheus — shared ROI for all GEE exports.
 * Must match configs/base.yaml (Nepal 1 km mask grid bbox).
 */
var minLon = 80.0129;
var minLat = 26.3386;
var maxLon = 88.2056;
var maxLat = 30.5158;
var roi = ee.Geometry.Rectangle([minLon, minLat, maxLon, maxLat], 'EPSG:4326', false);

// Pre-monsoon fire season (same as labels)
var START_YEAR = 2016;
var END_YEAR = 2026;
var MONTHS = [1, 2, 3, 4, 5];

exports.roi = roi;
exports.START_YEAR = START_YEAR;
exports.END_YEAR = END_YEAR;
exports.MONTHS = MONTHS;
exports.DRIVE_ROOT = 'Prometheus_GEE';  // folder in your Google Drive
