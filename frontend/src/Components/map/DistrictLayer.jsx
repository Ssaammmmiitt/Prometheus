import { useMemo } from "react";
import { GeoJSON } from "react-leaflet";

import { classFill } from "../../lib/riskColors";
import { useTheme } from "../../theme/ThemeProvider";

export default function DistrictLayer({
  geojson,
  horizon,
  selectedId,
  onDistrictClick,
  dimUnselected = false,
}) {
  const { theme } = useTheme();
  const stroke = useMemo(() => {
    const s = getComputedStyle(document.documentElement);
    return {
      accent: s.getPropertyValue("--accent").trim() || (theme === "dark" ? "#c8ff3a" : "#c2410c"),
      ink: s.getPropertyValue("--ink").trim() || (theme === "dark" ? "#eef0e6" : "#161513"),
    };
  }, [theme]);

  const style = useMemo(() => {
    return (feature) => {
      const props = feature.properties ?? {};
      const id = Number(props.district_id);
      const selected = selectedId != null && id === Number(selectedId);
      const faded = dimUnselected && selectedId != null && !selected;
      return {
        color: selected ? stroke.accent : `${stroke.ink}59`,
        weight: selected ? 2.5 : 1,
        fillColor: classFill(props.risk_class, selected ? 0.45 : faded ? 0.06 : 0.22),
        fillOpacity: 1,
      };
    };
  }, [selectedId, dimUnselected, stroke]);

  if (!geojson) return null;

  return (
    <GeoJSON
      key={`${geojson.features?.length ?? 0}-${horizon}-${selectedId ?? "all"}-${geojson.features?.[0]?.properties?.[`mean_h${horizon}`] ?? ""}`}
      data={geojson}
      style={style}
      onEachFeature={(feature, layer) => {
        const props = feature.properties ?? {};
        layer.bindTooltip(
          `${props.name ?? "District"} — ${props.risk_class_name ?? "unknown"} danger`,
          { sticky: true, className: "label-ui" },
        );
        layer.on("click", (e) => {
          if (onDistrictClick) onDistrictClick(feature, e.latlng);
        });
      }}
    />
  );
}
