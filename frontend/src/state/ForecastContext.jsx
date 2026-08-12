import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { getForecasts } from "../api/client";
import { DEFAULT_DATE } from "../lib/nepal";

const ForecastContext = createContext(null);

export function ForecastProvider({ children }) {
  const [params, setParams] = useSearchParams();
  const [catalog, setCatalog] = useState({
    dates: [],
    years: ["2024", "2025", "2026"],
    default_date: DEFAULT_DATE,
    horizons: [1, 7],
  });
  const [apiError, setApiError] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getForecasts()
      .then((body) => {
        if (cancelled) return;
        setCatalog(body);
        setApiError(null);
        setReady(true);
      })
      .catch((err) => {
        if (cancelled) return;
        setApiError(err.message);
        setReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const [liveDate, setLiveDate] = useState(null);
  const urlDate = params.get("date") || catalog.default_date || DEFAULT_DATE;
  const date = liveDate ?? urlDate;
  const horizon = Number(params.get("horizon")) === 7 ? 7 : 1;

  useEffect(() => {
    const known = catalog.dates ?? [];
    if (!known.length) return;
    if (!known.includes(date) && catalog.default_date) {
      const sp = new URLSearchParams(params);
      sp.set("date", catalog.default_date);
      sp.set("horizon", String(horizon));
      setParams(sp, { replace: true });
    }
  }, [catalog, date, horizon, params, setParams]);

  const patch = useCallback(
    (next) => {
      const sp = new URLSearchParams(params);
      Object.entries(next).forEach(([k, v]) => {
        if (v === undefined || v === null) sp.delete(k);
        else sp.set(k, String(v));
      });
      setParams(sp, { replace: true });
    },
    [params, setParams],
  );

  const setDate = useCallback(
    (d, opts = {}) => {
      if (opts.live) {
        setLiveDate(d);
        return;
      }
      setLiveDate(null);
      patch({ date: d, horizon });
    },
    [patch, horizon],
  );
  const setHorizon = useCallback(
    (h) => {
      setLiveDate(null);
      patch({ date, horizon: h });
    },
    [patch, date],
  );

  const query = useMemo(() => `date=${date}&horizon=${horizon}`, [date, horizon]);

  const value = useMemo(
    () => ({
      date,
      horizon,
      setDate,
      setHorizon,
      catalog,
      dates: catalog.dates ?? [],
      years: catalog.years ?? ["2024", "2025", "2026"],
      apiError,
      ready,
      query,
    }),
    [date, horizon, setDate, setHorizon, catalog, apiError, ready, query],
  );

  return <ForecastContext.Provider value={value}>{children}</ForecastContext.Provider>;
}

export function useForecast() {
  const ctx = useContext(ForecastContext);
  if (!ctx) throw new Error("useForecast must be used inside ForecastProvider");
  return ctx;
}
