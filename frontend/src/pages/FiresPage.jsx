import { useEffect, useState } from "react";

import { getActiveFires } from "../api/client";
import DateScrubber from "../components/map/DateScrubber";
import FirePointsLayer from "../components/map/FirePointsLayer";
import HorizonToggle from "../components/map/HorizonToggle";
import NepalBorderLayer from "../components/map/NepalBorderLayer";
import NepalMap from "../components/map/NepalMap";
import RiskTileLayer from "../components/map/RiskTileLayer";
import Card from "../components/ui/Card";
import { FlameGlyph } from "../lib/flameMark.js";
import { prettyDate } from "../lib/plain";
import { useForecast } from "../state/ForecastContext";

export default function FiresPage() {
  const { date, horizon, setDate, setHorizon, dates, years } = useForecast();
  const [lookback, setLookback] = useState(2);
  const [showRisk, setShowRisk] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [features, setFeatures] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getActiveFires({ asOf: date, lookbackDays: lookback, limit: 2000 })
      .then((body) => {
        if (!cancelled) {
          setFeatures(body.features ?? []);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFeatures([]);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [date, lookback]);

  const today = new Date().toISOString().slice(0, 10);
  const live = date === today && features.length > 0;

  return (
    <div className="absolute inset-0 pt-14">
      <NepalMap>
        {showRisk && <RiskTileLayer date={date} horizon={horizon} opacity={0.4} />}
        <NepalBorderLayer />
        <FirePointsLayer features={features} live={live} />
      </NepalMap>

      <Card className="absolute top-[4.75rem] left-2 md:left-4 z-900 w-[calc(100%-1rem)] md:w-72 p-3 md:p-4 panel-enter">
        <p className="label-ui text-muted">Satellite fire detections</p>
        <p className="font-display font-extrabold text-4xl tabular-nums mt-1">
          {loading ? "—" : features.length}
        </p>
        <p className="text-xs text-muted mt-2 leading-relaxed">
          Each <FlameGlyph size={12} /> is a satellite fire detection around{" "}
          {prettyDate(date)}
          {lookback > 1 ? `, plus the ${lookback - 1} day(s) before` : ""}.
          These are observations, not the forecast.
        </p>
        <div className="mt-3">
          <HorizonToggle value={horizon} onChange={setHorizon} />
        </div>
        <label className="block mt-3">
          <span className="text-sm text-ink">
            How many past days: {lookback}
          </span>
          <input
            type="range"
            min={1}
            max={7}
            value={lookback}
            onChange={(e) => setLookback(Number(e.target.value))}
            className="w-full mt-1 accent-[var(--accent)]"
          />
        </label>
        <label className="flex items-center gap-2 mt-2 min-h-11 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={showRisk}
            onChange={(e) => setShowRisk(e.target.checked)}
            className="accent-[var(--accent)] size-4"
          />
          Show danger colors underneath
        </label>
        {live && <p className="label-ui text-live mt-2">Live</p>}
        {!loading && features.length === 0 && (
          <p className="text-sm text-muted mt-2">
            No fires in this window — try another day or a longer lookback.
          </p>
        )}
      </Card>

      <DateScrubber
        date={date}
        dates={dates}
        years={years}
        onDate={setDate}
        playing={playing}
        onPlaying={setPlaying}
      />
    </div>
  );
}
