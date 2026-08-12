/** District class fills — same family as the COG yellow→purple ramp. */
export const CLASS_COLORS = {
  0: "#ffffcc",
  1: "#fecc66",
  2: "#fd8d3c",
  3: "#fc4e2a",
  4: "#5b21b6",
};

export const TILE_BINS = [
  { label: "0–5%", color: "#ffffcc" },
  { label: "5–10%", color: "#fecc66" },
  { label: "10–20%", color: "#fd8d3c" },
  { label: "20–40%", color: "#fc4e2a" },
  { label: "40–70%", color: "#bd0026" },
  { label: "70–100%", color: "#5b21b6" },
];

export const CLASS_NAMES = ["Low", "Moderate", "High", "Very High", "Extreme"];

export function classFill(riskClass, opacity = 0.28) {
  const hex = CLASS_COLORS[riskClass] ?? "#8a8f80";
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}
