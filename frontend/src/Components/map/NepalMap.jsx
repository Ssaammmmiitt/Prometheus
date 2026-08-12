import { MapContainer, TileLayer, useMapEvents } from "react-leaflet";

import { NEPAL_CENTER, NEPAL_LATLNG_BOUNDS } from "../../lib/nepal";
import { useTheme } from "../../theme/ThemeProvider";
import ZoomBar from "./ZoomBar";

function ClickCatch({ onClick }) {
  useMapEvents({
    click(e) {
      if (onClick) onClick(e.latlng);
    },
  });
  return null;
}

export default function NepalMap({ children, onClick, zoom = 7 }) {
  const { isDark } = useTheme();
  const tiles = isDark
    ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
    : "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";

  return (
    <MapContainer
      center={NEPAL_CENTER}
      zoom={zoom}
      minZoom={6}
      maxZoom={12}
      maxBounds={NEPAL_LATLNG_BOUNDS}
      className="h-full w-full"
      zoomControl={false}
      attributionControl
      scrollWheelZoom
      dragging
      touchZoom
      bounceAtZoomLimits={false}
    >
      <TileLayer
        key={isDark ? "dark" : "light"}
        url={tiles}
        attribution="&copy; OSM &copy; CARTO"
      />
      <ClickCatch onClick={onClick} />
      {children}
      <ZoomBar />
    </MapContainer>
  );
}
