import { LEGEND_BINS } from "../../lib/plain";
import Card from "../ui/Card";

export default function MapLegend() {
  return (
    <Card className="absolute bottom-[8.5rem] md:bottom-28 left-2 md:left-4 z-800 w-[min(100%-1rem,16rem)] p-3 md:p-4">
      <p className="label-ui text-muted mb-2">Chance of fire</p>
      <div className="flex h-2 w-full overflow-hidden mb-2">
        {LEGEND_BINS.map((bin) => (
          <span key={bin.label} className="flex-1" style={{ background: bin.color }} />
        ))}
      </div>
      <div className="flex justify-between text-[10px] uppercase tracking-wider text-muted">
        <span>Quiet</span>
        <span>Most dangerous</span>
      </div>
    </Card>
  );
}
