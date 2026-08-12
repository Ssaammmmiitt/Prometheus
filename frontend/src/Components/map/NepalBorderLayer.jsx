import { useEffect, useMemo, useState } from "react";
import { GeoJSON } from "react-leaflet";

import { useTheme } from "../../theme/ThemeProvider";

/**
 * National outline so Nepal stays readable on light basemaps
 * (CARTO light country lines are nearly invisible).
 */
export default function NepalBorderLayer() {
  const { isDark } = useTheme();
  const [geojson, setGeojson] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/nepal-border.geojson")
      .then((r) => r.json())
      .then((body) => {
        if (!cancelled) setGeojson(body);
      })
      .catch(() => {
        if (!cancelled) setGeojson(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const style = useMemo(
    () => ({
      color: isDark ? "#eef0e6" : "#161513",
      weight: isDark ? 1.5 : 2.25,
      opacity: isDark ? 0.7 : 0.95,
      fill: false,
      fillOpacity: 0,
      interactive: false,
    }),
    [isDark],
  );

  if (!geojson) return null;

  return (
    <GeoJSON
      key={isDark ? "nepal-dark" : "nepal-light"}
      data={geojson}
      style={style}
      interactive={false}
    />
  );
}
