import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CircleMarker } from "react-leaflet";

import { ApiError, getWhatIfSchema, postWhatIf } from "../api/client";
import DateScrubber from "../components/map/DateScrubber";
import HorizonToggle from "../components/map/HorizonToggle";
import NepalMap from "../components/map/NepalMap";
import RiskTileLayer from "../components/map/RiskTileLayer";
import Button from "../components/ui/Button";
import { CLASS_COLORS } from "../lib/riskColors";
import {
  FEATURE_LABELS,
  RISK_WORDS,
  featureLabel,
  prepareExplain,
  prettyDate,
} from "../lib/plain";
import { useForecast } from "../state/ForecastContext";

const GROUPS = [
  { id: "temperature", title: "Air temperature" },
  { id: "moisture", title: "Humidity" },
  { id: "rain", title: "Rain and dry spells" },
  { id: "wind", title: "Wind" },
  { id: "plants", title: "How green the plants are" },
  { id: "ground", title: "Ground heat" },
];

function fmtVal(feature, value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  if (feature === "rh" || feature.includes("days")) return n.toFixed(0);
  if (feature.startsWith("ndvi") || feature === "evi") return n.toFixed(2);
  if (feature.startsWith("precip") || feature.startsWith("wind")) return n.toFixed(1);
  return n.toFixed(1);
}

function fmtPct(p) {
  if (p == null || Number.isNaN(Number(p))) return "—";
  return `${(Number(p) * 100).toFixed(2)}%`;
}

function classWord(name) {
  return RISK_WORDS[name] ?? name ?? "—";
}

function pickSliders(sliders, features) {
  const next = {};
  for (const s of sliders) {
    if (features[s.feature] != null) next[s.feature] = Number(features[s.feature]);
  }
  return next;
}

export default function PredictPage() {
  const { date, horizon, setDate, setHorizon, dates, years, ready } = useForecast();
  const [schema, setSchema] = useState(null);
  const [picked, setPicked] = useState(null);
  const [sliders, setSliders] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const timer = useRef(null);
  const requestId = useRef(0);

  useEffect(() => {
    let cancelled = false;
    getWhatIfSchema()
      .then((body) => {
        if (!cancelled) setSchema(body);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const score = useCallback(
    (lat, lon, overrides, { immediate } = {}) => {
      if (timer.current) clearTimeout(timer.current);
      const run = () => {
        const id = ++requestId.current;
        setLoading(true);
        setError(null);
        postWhatIf({ lat, lon, date, horizon, overrides, top: 8 })
          .then((body) => {
            if (id !== requestId.current) return;
            setResult(body);
            setLoading(false);
          })
          .catch((err) => {
            if (id !== requestId.current) return;
            setResult(null);
            setError(err);
            setLoading(false);
          });
      };
      if (immediate) run();
      else timer.current = setTimeout(run, 180);
    },
    [date, horizon],
  );

  useEffect(() => () => timer.current && clearTimeout(timer.current), []);

    useEffect(() => {
    if (!picked || !schema) return undefined;
    setSliders({});
    setResult(null);
    score(picked.lat, picked.lon, {}, { immediate: true });
    return undefined;
  }, [picked, date, horizon, schema, score]);

  useEffect(() => {
    if (!result?.features || !schema) return;
    setSliders(pickSliders(schema.sliders, result.features));
  }, [result, schema]);

  const onMapClick = useCallback((ll) => {
    setPicked({ lat: ll.lat, lon: ll.lng });
    setSliders({});
    setResult(null);
  }, []);

  const onSlide = (feature, value) => {
    const next = { ...sliders, [feature]: value };
    setSliders(next);
    if (!picked) return;
    score(picked.lat, picked.lon, next);
  };

  const reset = () => {
    if (!picked || !result) return;
    const baseline = pickSliders(schema.sliders, result.features);
    // Re-fetch empty overrides so sliders snap to the real cell.
    setSliders({});
    score(picked.lat, picked.lon, {}, { immediate: true });
    void baseline;
  };

  const offMask =
    error instanceof ApiError &&
    typeof error.detail === "string" &&
    String(error.detail).toLowerCase().includes("forest mask");

  const scenario = result?.scenario;
  const baseline = result?.baseline;
  const factors = useMemo(() => prepareExplain(scenario?.top, 4), [scenario]);
  const color = CLASS_COLORS[scenario?.risk_class] ?? "#8a8f80";
  const slidersByGroup = useMemo(() => {
    const list = schema?.sliders ?? [];
    return GROUPS.map((g) => ({ ...g, items: list.filter((s) => s.group === g.id) })).filter(
      (g) => g.items.length,
    );
  }, [schema]);

  const vpd = result?.features?.vpd;
  const changed =
    baseline &&
    scenario &&
    Math.abs(scenario.probability - baseline.probability) > 1e-6;

  return (
    <div className="absolute inset-0 pt-[var(--app-header)] flex flex-col lg:flex-row min-h-0">
      <div className="relative min-w-0 h-[30vh] min-h-[11rem] sm:h-[34vh] lg:h-auto lg:flex-1">
        <NepalMap onClick={onMapClick} zoom={7}>
          {ready && dates.includes(date) && (
            <RiskTileLayer date={date} horizon={horizon} opacity={0.45} />
          )}
          {picked && (
            <CircleMarker
              center={[picked.lat, picked.lon]}
              radius={9}
              pathOptions={{
                color: "var(--accent)",
                fillColor: color,
                fillOpacity: 0.9,
                weight: 2,
              }}
            />
          )}
        </NepalMap>
        <DateScrubber
          date={date}
          dates={dates}
          years={years}
          onDate={setDate}
          playing={playing}
          onPlaying={setPlaying}
        />
      </div>

      <aside className="flex-1 min-h-0 lg:flex-none lg:w-[min(440px,42vw)] xl:w-[min(460px,38vw)] shrink-0 flex flex-col border-t lg:border-t-0 lg:border-l border-[var(--hairline)] bg-surface">
        <div className="shrink-0 z-950 sticky top-0 bg-surface/95 backdrop-blur-sm border-b border-[var(--hairline)] p-3 sm:p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="label-ui text-muted">What if the weather changed?</p>
              <h1 className="font-display font-extrabold text-2xl sm:text-3xl leading-none mt-1">
                Try a place
              </h1>
            </div>
          </div>
          <div className="mt-2">
            <HorizonToggle value={horizon} onChange={setHorizon} compact />
          </div>

          {scenario ? (
            <div className="mt-3 flex items-end justify-between gap-3">
              <div className="min-w-0">
                <p className="label-ui text-muted">
                  Chance · {prettyDate(date)}
                  {loading ? " · updating" : ""}
                </p>
                <p
                  className="font-display font-extrabold text-4xl sm:text-5xl tabular-nums leading-none mt-1"
                  style={{ color }}
                >
                  {fmtPct(scenario.probability)}
                </p>
                <p className="text-xs sm:text-sm mt-2 leading-snug">
                  {classWord(scenario.risk_class_name)}
                  {changed && (
                    <span className="text-muted">
                      {" "}
                      · was {fmtPct(baseline.probability)}
                    </span>
                  )}
                </p>
              </div>
              {picked && (
                <Button variant="outline" className="shrink-0 px-3" onClick={reset}>
                  Reset
                </Button>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted mt-3 leading-relaxed">
              {loading
                ? "Loading this place. The first click of a season can take ~15 s."
                : "Click a forest cell on the map. Then drag the sliders — this number stays here."}
            </p>
          )}

          {offMask && (
            <p className="text-sm text-muted mt-3 leading-relaxed">
              That click is not forest or grassland. Try a greener part of Nepal.
            </p>
          )}
          {!offMask && error && (
            <p className="text-sm mt-3 leading-relaxed">
              {error instanceof ApiError ? error.message : String(error)}
            </p>
          )}
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto overscroll-pane p-3 sm:p-4 lg:p-6 pb-[max(1rem,var(--app-bottom))]">
          <p className="text-sm text-muted leading-relaxed">
            Terrain and fire history stay those of the clicked cell. The number
            is a calibrated chance of a satellite fire detection — not a yes or
            no that something will burn.
            {result?.base_rate != null && (
              <>
                {" "}
                A typical forest day is about {fmtPct(result.base_rate)}.
              </>
            )}
            {vpd != null && (
              <>
                {" "}
                Air dryness (VPD): {Number(vpd).toFixed(2)} kPa, from
                temperature and humidity.
              </>
            )}
          </p>

          {result?.place && (
            <div className="mt-4">
              <p className="label-ui text-muted mb-2">This place (locked)</p>
              <ul className="text-xs text-muted space-y-1">
                {schema?.place_facts?.map((name) => (
                  <li key={name} className="flex justify-between gap-3">
                    <span>{FEATURE_LABELS[name] ?? featureLabel(name)}</span>
                    <span className="tabular-nums text-ink">
                      {fmtVal(name, result.place[name])}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {picked && schema && (
            <div className="mt-5 space-y-5">
              {slidersByGroup.map((group) => (
                <div key={group.id}>
                  <p className="label-ui text-muted mb-2">{group.title}</p>
                  <div className="space-y-3">
                    {group.items.map((s) => {
                      const value = sliders[s.feature];
                      if (value == null) return null;
                      return (
                        <label key={s.feature} className="block">
                          <span className="flex justify-between text-sm text-ink mb-1">
                            <span>{featureLabel(s.feature)}</span>
                            <span className="tabular-nums text-muted">
                              {fmtVal(s.feature, value)}
                              {s.unit ? ` ${s.unit}` : ""}
                            </span>
                          </span>
                          <input
                            type="range"
                            min={s.lo}
                            max={s.hi}
                            step={s.step}
                            value={value}
                            onChange={(e) => onSlide(s.feature, Number(e.target.value))}
                            className="w-full accent-[var(--accent)]"
                          />
                        </label>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}

          {factors.length > 0 && (
            <div className="mt-6">
              <p className="label-ui text-muted mb-2">What moved this score</p>
              <ul className="space-y-2">
                {factors.map((row) => (
                  <li key={row.key}>
                    <p className="text-sm leading-snug mb-1">
                      {row.label}
                      <span className="text-muted">
                        {` · ${row.pct}%`}
                        {row.up ? " · higher chance" : " · lower chance"}
                      </span>
                    </p>
                    <div className="h-2 w-full bg-[var(--hairline)]">
                      <span
                        className="block h-full"
                        style={{
                          width: `${row.pct}%`,
                          background: row.up ? "var(--accent)" : "var(--ink)",
                        }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
