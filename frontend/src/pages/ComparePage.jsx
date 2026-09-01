import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import DateScrubber from "../components/map/DateScrubber";
import NepalMap from "../components/map/NepalMap";
import RiskTileLayer from "../components/map/RiskTileLayer";
import Button from "../components/ui/Button";
import { useForecast } from "../state/ForecastContext";

export default function ComparePage() {
  const { date, horizon, dates, years, ready, query, setDate } = useForecast();
  const { t } = useTranslation();
  
  // Create a separate date state for the right map
  // Left map uses the global date from context
  const [rightDate, setRightDate] = useState(() => {
    if (!dates?.length) return date;
    // Default to the same date last year if possible
    const currentYear = date.substring(0, 4);
    const lastYear = String(Number(currentYear) - 1);
    const target = date.replace(currentYear, lastYear);
    return dates.includes(target) ? target : date;
  });

  // Handle right map date updates from the scrubber
  const [rightPlaying, setRightPlaying] = useState(false);
  const [leftPlaying, setLeftPlaying] = useState(false);

  // References for map syncing
  const map1Ref = useRef(null);
  const map2Ref = useRef(null);

  useEffect(() => {
    const map1 = map1Ref.current;
    const map2 = map2Ref.current;
    if (!map1 || !map2) return;

    let isSyncing1 = false;
    let isSyncing2 = false;

    const handleMap1Move = () => {
      if (!isSyncing1) {
        isSyncing2 = true;
        map2.setView(map1.getCenter(), map1.getZoom(), { animate: false });
        isSyncing2 = false;
      }
    };

    const handleMap2Move = () => {
      if (!isSyncing2) {
        isSyncing1 = true;
        map1.setView(map2.getCenter(), map2.getZoom(), { animate: false });
        isSyncing1 = false;
      }
    };

    map1.on("move", handleMap1Move);
    map2.on("move", handleMap2Move);

    return () => {
      map1.off("move", handleMap1Move);
      map2.off("move", handleMap2Move);
    };
  }, [map1Ref.current, map2Ref.current]);

  return (
    <div className="absolute inset-0 pt-[var(--app-header)] flex flex-col md:flex-row min-h-0 bg-surface">
      {/* Left Map */}
      <div className="relative flex-1 min-h-0 border-b md:border-b-0 md:border-r border-[var(--hairline)]">
        <div className="absolute top-2 left-2 lg:top-4 lg:left-4 z-900 pointer-events-none">
          <div className="bg-surface/95 backdrop-blur-sm border border-[var(--hairline)] rounded-md px-3 py-2 shadow-sm pointer-events-auto">
            <h2 className="font-display font-bold text-sm">Map 1</h2>
          </div>
        </div>
        
        <NepalMap mapRef={map1Ref} zoom={7}>
          {ready && dates.includes(date) && (
            <RiskTileLayer date={date} horizon={horizon} opacity={0.65} />
          )}
        </NepalMap>
        
        <DateScrubber
          date={date}
          dates={dates}
          years={years}
          onDate={setDate}
          playing={leftPlaying}
          onPlaying={setLeftPlaying}
        />
      </div>

      {/* Right Map */}
      <div className="relative flex-1 min-h-0">
        <div className="absolute top-2 left-2 lg:top-4 lg:left-4 z-900 pointer-events-none flex items-center gap-2">
          <div className="bg-surface/95 backdrop-blur-sm border border-[var(--hairline)] rounded-md px-3 py-2 shadow-sm pointer-events-auto">
            <h2 className="font-display font-bold text-sm">Map 2</h2>
          </div>
          <Link to={`/?${query}`} className="pointer-events-auto">
            <Button variant="outline" className="h-9 px-3 text-xs">
              Exit Compare
            </Button>
          </Link>
        </div>
        
        <NepalMap mapRef={map2Ref} zoom={7}>
          {ready && dates.includes(rightDate) && (
            <RiskTileLayer date={rightDate} horizon={horizon} opacity={0.65} />
          )}
        </NepalMap>
        
        <DateScrubber
          date={rightDate}
          dates={dates}
          years={years}
          onDate={setRightDate}
          playing={rightPlaying}
          onPlaying={setRightPlaying}
        />
      </div>
    </div>
  );
}
