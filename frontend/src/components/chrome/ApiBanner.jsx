import { useForecast } from "../../state/ForecastContext";

export default function ApiBanner() {
  const { apiError } = useForecast();
  if (!apiError) return null;
  return (
    <div className="absolute top-[var(--app-header)] inset-x-0 z-1000 px-4 py-2 bg-lift border-b border-[var(--hairline)]">
      <p className="text-sm text-live">
        Can't load the forecast. Start the server (`make api`) and refresh this page.
      </p>
    </div>
  );
}
