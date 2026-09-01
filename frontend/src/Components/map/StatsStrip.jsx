import { CLASS_NAMES } from "../../lib/riskColors";
import AnimatedNumber from "../ui/AnimatedNumber";
import Card from "../ui/Card";

export default function StatsStrip({ geojson }) {
  const feats = geojson?.features ?? [];
  const counts = { Extreme: 0, "Very High": 0, High: 0 };
  feats.forEach((f) => {
    const name = f.properties?.risk_class_name;
    if (name in counts) counts[name] += 1;
  });

  return (
    <Card className="p-3 md:p-4 panel-enter">
      <p className="label-ui text-muted mb-2">Districts in danger today</p>
      <div className="grid grid-cols-3 gap-3">
        {[
          ["Extreme", "Most"],
          ["Very High", "Serious"],
          ["High", "High"],
        ].map(([k, label]) => (
          <div key={k}>
            <p className="font-display font-extrabold text-2xl md:text-3xl leading-none tabular-nums">
              <AnimatedNumber value={counts[k]} />
            </p>
            <p className="label-ui text-muted mt-1">{label}</p>
          </div>
        ))}
      </div>
      <p className="sr-only">{CLASS_NAMES.join(" ")}</p>
    </Card>
  );
}
