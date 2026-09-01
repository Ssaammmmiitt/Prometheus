import { useMemo, useState } from "react";
import L from "leaflet";
import { Marker, useMap, useMapEvents } from "react-leaflet";

import { flameSvg } from "../../lib/flameMark.js";

function flameIcon(size, live) {
  return L.divIcon({
    className: "fire-mark",
    iconSize: [size, size],
    iconAnchor: [size / 2, size * 0.85],
    html: flameSvg(size, live),
  });
}

export default function FirePointsLayer({ features, live = false }) {
  const map = useMap();
  const [zoom, setZoom] = useState(() => map.getZoom());
  useMapEvents({
    zoomend: () => setZoom(map.getZoom()),
  });

  const size = zoom >= 9 ? 18 : zoom >= 7 ? 13 : 9;
  const icon = useMemo(() => flameIcon(size, live), [size, live]);

  if (!features?.length) return null;
  return (
    <>
      {features.map((f, i) => {
        const [lon, lat] = f.geometry.coordinates;
        const day = f.properties?.date ?? "";
        return (
          <Marker
            key={`${day}-${i}`}
            position={[lat, lon]}
            icon={icon}
            title={`Fire seen ${day}`}
            zIndexOffset={400}
          />
        );
      })}
    </>
  );
}
