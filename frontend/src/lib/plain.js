/** Plain-language copy so the map is usable without ML jargon. */

export function prettyDate(iso) {
  if (!iso || iso.length < 10) return iso ?? "—";
  const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

export const FEATURE_LABELS = {
  t2m_max: "Hottest temperature",
  t2m_min: "Coldest temperature",
  t2m: "Air temperature",
  d2m: "Dew point",
  precip: "Rain today",
  precip_7d: "Rain, last week",
  precip_30d: "Rain, last month",
  rh: "Humidity",
  rh_min_7d: "Driest humidity this week",
  vpd: "How dry the air is",
  wind_speed: "Wind",
  wind_max_7d: "Strongest wind this week",
  u10: "East–west wind",
  v10: "North–south wind",
  soil_water_l1: "Soil moisture",
  surface_pressure: "Air pressure",
  consecutive_dry_days: "Days without rain",
  days_since_rain: "Days since rain",
  ndvi: "How green the plants are",
  evi: "Plant greenness",
  ndvi_anomaly: "Plants vs a normal year",
  lst_day: "Ground heat, day",
  lst_night: "Ground heat, night",
  lst_diff: "Day–night ground heat gap",
  fire_clim: "Usual fire rate this date",
  days_since_fire: "Days since last fire",
  fires_1yr: "Fires in the last year",
  fires_3yr: "Fires in the last 3 years",
  fires_5yr: "Fires in the last 5 years",
  elevation: "Elevation",
  slope: "Steepness",
  aspect_sin: "Hillside direction",
  aspect_cos: "Hillside direction",
  twi: "How wet the slope stays",
  tree_frac: "Tree cover",
  shrub_frac: "Shrub cover",
  grass_frac: "Grass cover",
  crop_frac: "Crops",
  dist_road: "Distance to a road",
  dist_settlement: "Distance to a village",
  built_frac: "Built-up land",
  doy_sin: "Time of year",
  doy_cos: "Time of year",
};

export function featureLabel(name) {
  return FEATURE_LABELS[name] ?? name.replaceAll("_", " ");
}

/** Collapse lookalike SHAP rows (sin/cos, collinear twins) into one story. */
const EXPLAIN_GROUPS = {
  t2m_max: "heat",
  t2m: "heat",
  t2m_min: "heat",
  lst_day: "heat",
  lst_night: "heat",
  lst_diff: "heat",
  doy_sin: "season",
  doy_cos: "season",
  u10: "wind",
  v10: "wind",
  wind_speed: "wind",
  wind_max_7d: "wind",
  fire_clim: "fire_hist",
  days_since_fire: "fire_hist",
  fires_1yr: "fire_hist",
  fires_3yr: "fire_hist",
  fires_5yr: "fire_hist",
  rh: "moisture",
  rh_min_7d: "moisture",
  vpd: "moisture",
  d2m: "moisture",
  precip: "rain",
  precip_7d: "rain",
  precip_30d: "rain",
  consecutive_dry_days: "rain",
  days_since_rain: "rain",
  ndvi: "green",
  evi: "green",
  ndvi_anomaly: "green",
  aspect_sin: "aspect",
  aspect_cos: "aspect",
  surface_pressure: "elev",
  elevation: "elev",
};

const GROUP_LABELS = {
  heat: "Heat",
  season: "Time of year",
  wind: "Wind",
  fire_hist: "Fire history",
  moisture: "Humidity and dry air",
  rain: "Rain and dry spell",
  green: "How green the plants are",
  aspect: "Hillside direction",
  elev: "Elevation",
};

export function prepareExplain(rows, limit = 4) {
  const buckets = new Map();
  for (const row of rows ?? []) {
    const group = EXPLAIN_GROUPS[row.feature];
    const key = group ?? featureLabel(row.feature);
    const label = group ? GROUP_LABELS[group] : featureLabel(row.feature);
    const shap = Number(row.shap_value) || 0;
    const prev = buckets.get(key);
    if (!prev) {
      buckets.set(key, { key, label, shap });
    } else {
      prev.shap += shap;
    }
  }
  const merged = [...buckets.values()].map((row) => ({
    ...row,
    abs: Math.abs(row.shap),
    up: row.shap >= 0,
  }));
  merged.sort((a, b) => b.abs - a.abs);
  const top = merged.filter((row) => row.abs > 0).slice(0, limit);
  const total = merged.reduce((sum, row) => sum + row.abs, 0) || 1;
  return top.map((row) => ({
    ...row,
    share: row.abs / total,
    pct: Math.max(4, Math.round((row.abs / total) * 100)),
  }));
}

export const RISK_WORDS = {
  Low: "Quiet",
  Moderate: "Watch",
  High: "Elevated",
  VeryHigh: "Serious",
  "Very High": "Serious",
  Extreme: "Most dangerous",
};

export const LEGEND_BINS = [
  { label: "Quiet", hint: "little chance", color: "#ffffcc" },
  { label: "Low", hint: "", color: "#fecc66" },
  { label: "Watch", hint: "", color: "#fd8d3c" },
  { label: "High", hint: "", color: "#fc4e2a" },
  { label: "Serious", hint: "", color: "#bd0026" },
  { label: "Most dangerous", hint: "", color: "#5b21b6" },
];
