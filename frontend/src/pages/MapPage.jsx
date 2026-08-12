import { useCallback, useEffect, useState } from "react";
import { Layers } from "lucide-react";

import { getActiveFires, getDistricts } from "../api/client";
import DateScrubber from "../components/map/DateScrubber";
import DistrictLayer from "../components/map/DistrictLayer";
import ExplainDrawer from "../components/map/ExplainDrawer";
import FirePointsLayer from "../components/map/FirePointsLayer";
import HorizonToggle from "../components/map/HorizonToggle";
import MapLegend from "../components/map/MapLegend";
import NepalMap from "../components/map/NepalMap";
import RiskTileLayer from "../components/map/RiskTileLayer";
import StatsStrip from "../components/map/StatsStrip";
import Card from "../components/ui/Card";
import { FlameGlyph } from "../lib/flameMark.js";
import { useForecast } from "../state/ForecastContext";

export default function MapPage() {
  const { date, horizon, setDate, setHorizon, dates, years, query } = useForecast();
  const [geojson, setGeojson] = useState(null);
  const [geoError, setGeoError] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [showRisk, setShowRisk] = useState(true);
  const [showDistricts, setShowDistricts] = useState(true);
  const [showFires, setShowFires] = useState(false);
  const [opacity, setOpacity] = useState(0.65);
  const [fires, setFires] = useState([]);
  const [explain, setExplain] = useState(null);
  const [layersOpen, setLayersOpen] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(min-width: 768px)").matches,
  );

  useEffect(() => {
    let cancelled = false;
    const wait = playing ? 800 : 0;
    const handle = setTimeout(() => {
      getDistricts({ date, horizon })
        .then((body) => {
          if (!cancelled) {
            setGeojson(body);
            setGeoError(null);
          }
        })
        .catch((err) => {
          if (!cancelled) {
            setGeojson((prev) => {
              if (prev) return prev;
              setGeoError(err.message);
              return null;
            });
          }
        });
    }, wait);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [date, horizon, playing]);

  useEffect(() => {
    if (!showFires) {
      setFires([]);
      return undefined;
    }
    let cancelled = false;
    getActiveFires({ asOf: date, lookbackDays: 2 })
      .then((body) => {
        if (!cancelled) setFires(body.features ?? []);
      })
      .catch(() => {
        if (!cancelled) setFires([]);
      });
    return () => {
      cancelled = true;
    };
  }, [date, showFires]);

  const onDistrictClick = useCallback((feature, latlng) => {
    setExplain({
      lat: latlng.lat,
      lon: latlng.lng,
      date,
      horizon,
      district: {
        district_id: feature.properties.district_id,
        name: feature.properties.name,
      },
    });
  }, [date, horizon]);

  return (
    <div className="absolute inset-0 pt-14">
      <NepalMap
        onClick={(ll) =>
          setExplain({ lat: ll.lat, lon: ll.lng, date, horizon, district: null })
        }
      >
        {showRisk && !geoError && (
          <RiskTileLayer date={date} horizon={horizon} opacity={opacity} />
        )}
        {showDistricts && geojson && (
          <DistrictLayer
            geojson={geojson}
            horizon={horizon}
            onDistrictClick={onDistrictClick}
          />
        )}
        {showFires && <FirePointsLayer features={fires} />}
      </NepalMap>

      <div className="absolute top-[4.75rem] left-2 md:left-4 z-900 flex flex-col gap-2 w-[calc(100%-4.5rem)] md:w-72">
        <StatsStrip geojson={geojson} />
        <Card className="p-3 md:p-4">
          <button
            type="button"
            className="md:hidden flex items-center justify-between w-full min-h-11 label-ui text-ink"
            onClick={() => setLayersOpen((v) => !v)}
          >
            <span className="inline-flex items-center gap-2">
              <Layers size={14} /> Layers
            </span>
            <span>{layersOpen ? "Hide" : "Show"}</span>
          </button>
          <div className={`${layersOpen ? "block" : "hidden"} md:block mt-2 md:mt-0`}>
            <p className="hidden md:block text-xs text-muted mb-3 leading-relaxed">
              Yellow is quieter. Purple is the most dangerous forest for the
              chosen day.
            </p>
            <HorizonToggle value={horizon} onChange={setHorizon} />
            <div className="mt-3 space-y-2">
              <label className="flex items-center gap-2 min-h-11 text-sm text-ink cursor-pointer">
                <input
                  type="checkbox"
                  checked={showRisk}
                  onChange={(e) => setShowRisk(e.target.checked)}
                  className="accent-[var(--accent)] size-4"
                />
                Danger colors
              </label>
              {showRisk && (
                <div className="pl-6">
                  <p className="text-xs text-muted mb-1">
                    Color strength {Math.round(opacity * 100)}%
                  </p>
                  <input
                    type="range"
                    min={0.2}
                    max={1}
                    step={0.05}
                    value={opacity}
                    onChange={(e) => setOpacity(Number(e.target.value))}
                    className="w-full accent-[var(--accent)]"
                  />
                </div>
              )}
              <label className="flex items-center gap-2 min-h-11 text-sm text-ink cursor-pointer">
                <input
                  type="checkbox"
                  checked={showDistricts}
                  onChange={(e) => setShowDistricts(e.target.checked)}
                  className="accent-[var(--accent)] size-4"
                />
                District borders
              </label>
              <label className="flex items-center gap-2 min-h-11 text-sm text-ink cursor-pointer">
                <input
                  type="checkbox"
                  checked={showFires}
                  onChange={(e) => setShowFires(e.target.checked)}
                  className="accent-[var(--accent)] size-4"
                />
                <FlameGlyph size={13} />
                Recent fires
              </label>
            </div>
            <p className="text-xs text-muted mt-3 leading-relaxed hidden md:block">
              Tap a district to see why it is risky, then open its page.
            </p>
          </div>
        </Card>
      </div>

      <MapLegend />

      {geoError && (
        <Card className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-900 max-w-sm mx-4">
          <p className="label-ui text-muted mb-2">No map for this day</p>
          <p className="text-sm leading-relaxed">
            Forecasts cover January–May for years that were written to disk
            (right now 2024 and 2025). 2026 is in the training data — run
            `python scripts/forecast.py --backfill 2026` to put it on this map.
          </p>
        </Card>
      )}

      {explain && (
        <ExplainDrawer
          lat={explain.lat}
          lon={explain.lon}
          date={explain.date ?? date}
          horizon={explain.horizon ?? horizon}
          query={query}
          district={explain.district}
          onClose={() => setExplain(null)}
        />
      )}

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
