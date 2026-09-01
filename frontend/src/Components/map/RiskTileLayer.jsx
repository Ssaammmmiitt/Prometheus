import { useEffect, useRef, useState } from "react";
import { TileLayer } from "react-leaflet";

import { riskTileUrl } from "../../api/client";
import { NEPAL_LATLNG_BOUNDS } from "../../lib/nepal";

function sameDay(a, b) {
  return a?.date === b?.date && a?.horizon === b?.horizon;
}

/**
 * Keep the last fully loaded day on screen until the next day's tiles
 * finish, so play-back does not flash an empty map.
 */
export default function RiskTileLayer({ date, horizon, opacity = 0.65 }) {
  const [shown, setShown] = useState({ date, horizon });
  const [pending, setPending] = useState(null);
  const pendingRef = useRef(null);

  const adopt = (next) => {
    if (!next) return;
    setShown(next);
    setPending(null);
    pendingRef.current = null;
  };

  useEffect(() => {
    const next = { date, horizon };
    if (sameDay(next, shown)) {
      pendingRef.current = null;
      setPending(null);
      return;
    }
    pendingRef.current = next;
    setPending(next);
  }, [date, horizon, shown]);

  useEffect(() => {
    if (!pending) return undefined;
    const snapshot = pending;
    const failSafe = setTimeout(() => {
      if (sameDay(pendingRef.current, snapshot)) adopt(snapshot);
    }, 1800);
    return () => clearTimeout(failSafe);
  }, [pending]);

  return (
    <>
      <TileLayer
        key={`shown-${shown.date}-${shown.horizon}`}
        url={riskTileUrl(shown)}
        opacity={opacity}
        maxZoom={12}
        bounds={NEPAL_LATLNG_BOUNDS}
        keepBuffer={6}
        updateWhenZooming={false}
        zIndex={410}
        attribution="Prometheus risk"
        className="transition-opacity duration-500 ease-in-out"
      />
      {pending && (
        <TileLayer
          key={`pending-${pending.date}-${pending.horizon}`}
          url={riskTileUrl(pending)}
          opacity={0.01}
          maxZoom={12}
          bounds={NEPAL_LATLNG_BOUNDS}
          keepBuffer={6}
          updateWhenZooming={false}
          zIndex={411}
          className="transition-opacity duration-500 ease-in-out"
          eventHandlers={{
            load: (e) => {
              // Fade in the new layer
              e.target.setOpacity(opacity);
              
              // Wait for the transition to complete before making it the shown layer
              setTimeout(() => {
                if (sameDay(pendingRef.current, pending)) adopt(pending);
              }, 500);
            },
          }}
        />
      )}
    </>
  );
}
