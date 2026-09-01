import { useMap } from "react-leaflet";
import { Home, Minus, Plus } from "lucide-react";

import { NEPAL_CENTER } from "../../lib/nepal";

export default function ZoomBar() {
  const map = useMap();
  const btn =
    "min-h-11 min-w-11 flex items-center justify-center text-ink hover:bg-[var(--accent-12)] transition-colors duration-200 border-b border-[var(--hairline)] last:border-b-0";
  return (
    <div className="absolute top-2 right-2 md:top-4 md:right-4 z-800 flex flex-col bg-lift border border-[var(--hairline)]">
      <button type="button" className={btn} title="Zoom in" onClick={() => map.zoomIn()}>
        <Plus size={16} />
      </button>
      <button type="button" className={btn} title="Zoom out" onClick={() => map.zoomOut()}>
        <Minus size={16} />
      </button>
      <button
        type="button"
        className={btn}
        title="Show all of Nepal"
        onClick={() => map.flyTo(NEPAL_CENTER, 7, { duration: 0.6 })}
      >
        <Home size={16} />
      </button>
    </div>
  );
}
