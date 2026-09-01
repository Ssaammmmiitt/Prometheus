export const NEPAL_BOUNDS = {
  north: 30.45,
  south: 26.35,
  east: 88.2,
  west: 80.05,
};

export const NEPAL_CENTER = [28.3949, 84.124];

export const NEPAL_LATLNG_BOUNDS = [
  [NEPAL_BOUNDS.south, NEPAL_BOUNDS.west],
  [NEPAL_BOUNDS.north, NEPAL_BOUNDS.east],
];

export const DEFAULT_DATE = "2026-04-12";
export const FORECAST_YEARS = ["2026", "2025", "2024"];
export const SEASON_START = "01-01";
export const SEASON_END = "05-31";
export const LOYO_PR_AUC = 0.1548;

export function seasonBounds(dateStr) {
  const year = String(dateStr).slice(0, 4);
  return { start: `${year}-01-01`, end: `${year}-05-31`, year };
}

export function datesForYear(dates, year) {
  const prefix = String(year);
  return dates.filter((d) => d.startsWith(prefix));
}
