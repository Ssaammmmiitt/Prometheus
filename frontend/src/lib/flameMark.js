import { createElement } from "react";
import { Flame } from "lucide-react";

/** Lucide Flame path — filled so it stays readable at map-marker size. */
const FLAME_D =
  "M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z";

export function flameSvg(size, live = false) {
  const cls = live ? "fire-pip is-live" : "fire-pip";
  return `<svg class="${cls}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path d="${FLAME_D}" fill="var(--live)" stroke="var(--surface)" stroke-width="1.75" stroke-linejoin="round"/>
  </svg>`;
}

export function FlameGlyph({ size = 14, live = false, className = "" }) {
  return createElement(Flame, {
    size,
    strokeWidth: 2,
    fill: "currentColor",
    className: `inline-block align-[-2px] text-live ${live ? "fire-pip is-live" : ""} ${className}`.trim(),
    "aria-hidden": "true",
  });
}
