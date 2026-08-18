import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getDistricts, getDistrictTimeseries } from "../api/client";
import DistrictTimeseries from "../components/charts/DistrictTimeseries";
import DistrictLayer from "../components/map/DistrictLayer";
import HorizonToggle from "../components/map/HorizonToggle";
import NepalMap from "../components/map/NepalMap";
import RiskTileLayer from "../components/map/RiskTileLayer";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";
import { seasonBounds } from "../lib/nepal";
import { prettyDate } from "../lib/plain";
import { useForecast } from "../state/ForecastContext";

function pct(v) {
  if (typeof v !== "number" || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export default function DistrictPage() {
  const { id } = useParams();
  const { date, horizon, setHorizon, query, dates, ready } = useForecast();
  const [geojson, setGeojson] = useState(null);
  const [series, setSeries] = useState([]);
  const [error, setError] = useState(null);
  const season = seasonBounds(date);

  useEffect(() => {
    if (!ready || !dates.includes(date)) return undefined;
    let cancelled = false;
    getDistricts({ date, horizon })
      .then((body) => {
        if (!cancelled) setGeojson(body);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [date, dates, horizon, ready]);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    getDistrictTimeseries({
      id,
      horizon,
      start: season.start,
      end: season.end,
    })
      .then((body) => {
        if (!cancelled) setSeries(body.timeseries ?? []);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [id, horizon, season.start, season.end]);

  const feature = geojson?.features?.find(
    (f) => Number(f.properties?.district_id) === Number(id),
  );
  const props = feature?.properties ?? {};
  const mean = props[`mean_h${horizon}`];
  const max = props[`max_h${horizon}`];

  return (
    <div className="absolute inset-0 pt-14 flex flex-col md:flex-row">
      <div className="h-[38vh] md:h-auto md:flex-1 relative min-w-0">
        <NepalMap zoom={7}>
          {ready && dates.includes(date) && (
            <RiskTileLayer date={date} horizon={horizon} opacity={0.45} />
          )}
          {geojson && (
            <DistrictLayer
              geojson={geojson}
              horizon={horizon}
              selectedId={id}
              dimUnselected
            />
          )}
        </NepalMap>
      </div>
      <aside className="flex-1 md:flex-none md:w-[min(420px,42vw)] shrink-0 border-t md:border-t-0 md:border-l border-[var(--hairline)] bg-surface overflow-y-auto p-4 md:p-6">
        <Link to={`/?${query}`}>
          <Button variant="ghost" className="px-0">
            ← Back to Nepal
          </Button>
        </Link>
        <p className="label-ui text-muted mt-4">One district</p>
        <h1 className="font-display font-extrabold text-3xl md:text-4xl leading-none mt-1">
          {props.name ?? "—"}
        </h1>
        <p className="text-sm mt-3">
          On {prettyDate(date)} this district is{" "}
          <span className="text-accent font-semibold">
            {props.risk_class_name ?? "unscored"}
          </span>{" "}
          danger
          {horizon === 7 ? " over the next week" : " for tomorrow"}.
        </p>

        <div className="grid grid-cols-2 gap-3 mt-5">
          <Card className="p-4">
            <p className="label-ui text-muted">Typical chance</p>
            <p className="font-display font-extrabold text-3xl tabular-nums mt-1">
              {pct(mean)}
            </p>
            <p className="text-[11px] text-muted mt-1">Average forest cell</p>
          </Card>
          <Card className="p-4">
            <p className="label-ui text-muted">Hottest cell</p>
            <p className="font-display font-extrabold text-3xl tabular-nums mt-1">
              {pct(max)}
            </p>
            <p className="text-[11px] text-muted mt-1">Highest in the district</p>
          </Card>
        </div>

        <div className="mt-5">
          <HorizonToggle value={horizon} onChange={setHorizon} />
        </div>

        <p className="label-ui text-muted mt-6 mb-2">How it changed this season</p>
        {error && <p className="text-sm text-muted">{error}</p>}
        <DistrictTimeseries rows={series} selectedDate={date} />
        <p className="text-xs text-muted mt-3 leading-relaxed">
          The line is the district’s average fire chance. The highlighted point
          is the day on the map.
        </p>
      </aside>
    </div>
  );
}
