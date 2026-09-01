import { useCallback, useEffect, useState } from "react";
import { Layers } from "lucide-react";
import { useTranslation } from "react-i18next";

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
  const { date, horizon, setDate, setHorizon, dates, years, ready } = useForecast();
  const { t } = useTranslation();
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
    () => typeof window !== "undefined" && window.matchMedia("(min-width: 1024px)").matches,
  );

  useEffect(() => {
    if (!ready || !dates.includes(date)) return undefined;
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
  }, [date, dates, horizon, playing, ready]);

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
        mean: feature.properties[`mean_h${horizon}`],
        max: feature.properties[`max_h${horizon}`],
        n_forest: feature.properties.n_forest_cells,
        class_name: feature.properties.risk_class_name,
      },
    });
  }, [date, horizon]);

  return (
    <div className="absolute inset-0 pt-[var(--app-header)]">
      <NepalMap
        onClick={(ll) =>
          setExplain({ lat: ll.lat, lon: ll.lng, date, horizon, district: null })
        }
      >
        {showRisk && ready && dates.includes(date) && !geoError && (
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

      <div
        className={`absolute top-2 left-2 lg:top-4 lg:left-4 z-900 flex flex-col gap-2 w-[min(20rem,calc(100%-4.5rem))] lg:w-72 ${
          explain ? "hidden lg:flex" : ""
        }`}
      >
        <StatsStrip geojson={geojson} />
        <Card className="p-3 md:p-4">
          <button
            type="button"
            className="lg:hidden flex items-center justify-between w-full min-h-11 label-ui text-ink"
            onClick={() => setLayersOpen((v) => !v)}
          >
            <span className="inline-flex items-center gap-2">
              <Layers size={14} /> {t("map.layers")}
            </span>
            <span>{layersOpen ? t("map.hide") : t("map.show")}</span>
          </button>
          <div className={`${layersOpen ? "block" : "hidden"} lg:block mt-2 lg:mt-0`}>
            <p className="hidden lg:block text-xs text-muted mb-3 leading-relaxed">
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
                {t("map.dangerColors")}
              </label>
              {showRisk && (
                <div className="pl-6">
                  <p className="text-xs text-muted mb-1">
                    {t("map.colorStrength", { pct: Math.round(opacity * 100) })}
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
                {t("map.districtBorders")}
              </label>
              <label className="flex items-center gap-2 min-h-11 text-sm text-ink cursor-pointer">
                <input
                  type="checkbox"
                  checked={showFires}
                  onChange={(e) => setShowFires(e.target.checked)}
                  className="accent-[var(--accent)] size-4"
                />
                <FlameGlyph size={13} />
                {t("map.recentFires")}
              </label>
            </div>
            <p className="text-xs text-muted mt-3 leading-relaxed hidden lg:block">
              Tap a district to see why it is risky, then open its page.
            </p>
          </div>
        </Card>
      </div>

      <div className={explain ? "hidden lg:block" : ""}>
        <MapLegend />
      </div>

      {geoError && (
        <Card className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-900 max-w-sm mx-4">
          <p className="label-ui text-muted mb-2">{t("map.noMap")}</p>
          <p className="text-sm leading-relaxed">
            {t("map.noMapDesc")}
          </p>
        </Card>
      )}

      {(!ready || !dates.includes(date)) && !geoError && (
        <Card className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-900 max-w-sm mx-4">
          <p className="label-ui text-muted mb-2">{t("map.loading")}</p>
          <p className="text-sm leading-relaxed">
            {t("map.loadingDesc")}
          </p>
        </Card>
      )}

      {explain && (
        <ExplainDrawer
          lat={explain.lat}
          lon={explain.lon}
          district={explain.district}
          onClose={() => setExplain(null)}
        />
      )}

      <div className={explain ? "hidden lg:block" : ""}>
        <DateScrubber
          date={date}
          dates={dates}
          years={years}
          onDate={setDate}
          playing={playing}
          onPlaying={setPlaying}
        />
      </div>
    </div>
  );
}
