import { MapContainer, TileLayer, useMap, useMapEvents } from "react-leaflet";
import { useEffect } from "react";

import { NEPAL_CENTER, NEPAL_LATLNG_BOUNDS } from "../../lib/nepal";
import { useTheme } from "../../theme/ThemeProvider";
import NepalBorderLayer from "./NepalBorderLayer";
import ZoomBar from "./ZoomBar";

function ClickCatch({ onClick }) {
  useMapEvents({
    click(e) {
      if (onClick) onClick(e.latlng);
    },
  });
  return null;
}

function InvalidateOnResize() {
  const map = useMap();
  useEffect(() => {
    const bump = () => map.invalidateSize();
    const t = window.setTimeout(bump, 250);
    window.addEventListener("resize", bump);
    window.addEventListener("orientationchange", bump);
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("resize", bump);
      window.removeEventListener("orientationchange", bump);
    };
  }, [map]);
  return null;
}

export default function NepalMap({ children, onClick, zoom = 7, mapRef }) {
  const { isDark } = useTheme();
  const baseTiles = isDark
    ? "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
    : "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}";
    
  const referenceTiles = isDark
    ? "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
    : "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}";

  return (
    <MapContainer
      ref={mapRef}
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
        key={isDark ? "dark-base" : "light-base"}
        url={baseTiles}
        attribution="Tiles &copy; Esri"
      />
      <TileLayer
        key={isDark ? "dark-ref" : "light-ref"}
        url={referenceTiles}
      />
      <ClickCatch onClick={onClick} />
      <InvalidateOnResize />
      {children}
      <NepalBorderLayer />
      <ZoomBar />
    </MapContainer>
  );
}
