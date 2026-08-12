export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

function qs(params) {
  const sp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  });
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export async function apiGet(path, params = {}) {
  const url = `${path}${qs(params)}`;
  let resp;
  try {
    resp = await fetch(url);
  } catch {
    throw new ApiError(0, "API unreachable — start the backend with `make api`.");
  }
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail ?? body;
    } catch {
      /* ignore */
    }
    throw new ApiError(resp.status, detail);
  }
  return resp.json();
}

export function getHealth() {
  return apiGet("/api/health");
}

export function getForecasts() {
  return apiGet("/api/forecasts");
}

export function getDistricts({ date, horizon }) {
  return apiGet("/api/districts", { date, horizon });
}

export function getDistrictTimeseries({ id, horizon, start, end }) {
  return apiGet(`/api/districts/${id}/timeseries`, { horizon, start, end });
}

export function getActiveFires({ asOf, lookbackDays = 2, limit = 1500 }) {
  return apiGet("/api/fires/active", {
    as_of: asOf,
    lookback_days: lookbackDays,
    limit,
  });
}

export function getVerification({ start, end } = {}) {
  return apiGet("/api/verification", { start, end });
}

export function getExplain({ lat, lon, date, horizon, top = 6 }) {
  return apiGet("/api/explain", { lat, lon, date, horizon, top });
}

export function riskTileUrl({ date, horizon }) {
  return `/api/risk/tiles/{z}/{x}/{y}.png?date=${date}&horizon=${horizon}`;
}
