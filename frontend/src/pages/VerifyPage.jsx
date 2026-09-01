import { useEffect, useMemo, useState } from "react";

import { getVerification } from "../api/client";
import { VerificationSparkline } from "../components/charts/VerificationSparkline";
import Card from "../components/ui/Card";
import { prettyDate } from "../lib/plain";
import { useForecast } from "../state/ForecastContext";

function fmt(v, digits = 3) {
  if (v == null || v === "" || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(digits);
}

function fmtPct(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${(Number(v) * 100).toFixed(1)}%`;
}

export default function VerifyPage() {
  const { date, setDate } = useForecast();
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getVerification()
      .then((body) => {
        if (!cancelled) setPayload(body);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const rows = useMemo(() => payload?.rows ?? [], [payload]);
  const summary = payload?.summary ?? {};
  const selected = useMemo(
    () => rows.find((r) => r.forecast_date === date) ?? null,
    [rows, date],
  );

  return (
    <div className="absolute inset-0 pt-[var(--app-header)] overflow-y-auto overscroll-pane">
      <div className="max-w-6xl mx-auto px-4 md:px-6 py-6 md:py-10 pb-[max(1.5rem,var(--app-bottom))]">
        <p className="label-ui text-muted">Did the map work?</p>
        <h1 className="font-display font-extrabold text-3xl md:text-5xl leading-none mt-2">
          Checking the forecast
        </h1>
        <p className="text-sm text-muted max-w-[62ch] mt-4 leading-relaxed">
          Checking the forecast against the next day’s satellites, for every
          mapped day in 2024–2026. A quiet day with almost no fire is shown as
          a dash — not a fake zero.
        </p>

        {error && <p className="text-sm mt-6">{error}</p>}

        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 md:gap-4 mt-6">
          <Card className="p-4">
            <p className="label-ui text-muted">Days checked</p>
            <p className="font-display font-extrabold text-3xl md:text-4xl tabular-nums mt-1">
              {summary.days ?? "—"}
            </p>
          </Card>
          <Card className="p-4">
            <p className="label-ui text-muted">Ranking skill</p>
            <p className="font-display font-extrabold text-3xl md:text-4xl tabular-nums mt-1">
              {fmt(summary.mean_pr_auc)}
            </p>
            <p className="text-[11px] text-muted mt-1">
              Higher is better. 0.08–0.25 is typical here.
            </p>
          </Card>
          <Card className="p-4">
            <p className="label-ui text-muted">Caught in hottest 10%</p>
            <p className="font-display font-extrabold text-3xl md:text-4xl tabular-nums mt-1">
              {fmtPct(summary.mean_top10_capture)}
            </p>
            <p className="text-[11px] text-muted mt-1">
              Share of real fires that sat in the reddest tenth of the map.
            </p>
          </Card>
          <Card className="p-4">
            <p className="label-ui text-muted">Fractions Skill Score</p>
            <p className="font-display font-extrabold text-3xl md:text-4xl tabular-nums mt-1">
              {fmt(summary.mean_fss)}
            </p>
            <p className="text-[11px] text-muted mt-1">
              Spatial accuracy of the forecast.
            </p>
          </Card>
          <Card className="p-4">
            <p className="label-ui text-muted">Economic Value (REV)</p>
            <p className="font-display font-extrabold text-3xl md:text-4xl tabular-nums mt-1">
              {fmt(summary.mean_rev)}
            </p>
            <p className="text-[11px] text-muted mt-1">
              Value saved relative to climatology.
            </p>
          </Card>
        </div>

        <Card className="mt-6 p-4 md:p-6">
          <p className="label-ui text-muted mb-3">Skill by day — tap a point</p>
          <VerificationSparkline rows={rows} selectedDate={date} onSelect={setDate} />
        </Card>

        {selected && (
          <p className="text-sm text-muted mt-5">
            {prettyDate(selected.forecast_date)}: satellites saw{" "}
            <span className="text-ink">{selected.n_pos}</span> fire cells the
            next day ({fmtPct(selected.base_rate)} of forest).
          </p>
        )}

        <div className="mt-4 overflow-x-auto border border-[var(--hairline)] -mx-4 md:mx-0">
          <table className="w-full text-left text-xs md:text-[12px] min-w-[640px]">
            <thead className="text-muted border-b border-[var(--hairline)]">
              <tr>
                {[
                  "Map date",
                  "Fire seen on",
                  "Fires",
                  "% of forest",
                  "Ranking skill",
                  "Caught in top 10%",
                  "Chance error",
                  "FSS",
                  "Econ. Value (REV)",
                ].map((h) => (
                  <th key={h} className="px-3 py-3 font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const active = r.forecast_date === date;
                const quiet = r.valid === false;
                return (
                  <tr
                    key={r.forecast_date}
                    onClick={() => setDate(r.forecast_date)}
                    className={`border-b border-[var(--hairline)] cursor-pointer min-h-11 hover:bg-[var(--accent-12)] ${
                      active ? "bg-[var(--accent-12)]" : ""
                    }`}
                  >
                    <td className="px-3 py-3 tabular-nums">{prettyDate(r.forecast_date)}</td>
                    <td className="px-3 py-3 tabular-nums">{prettyDate(r.observe_date)}</td>
                    <td className="px-3 py-3 tabular-nums">{r.n_pos}</td>
                    <td className="px-3 py-3 tabular-nums">{fmtPct(r.base_rate)}</td>
                    <td className="px-3 py-3 tabular-nums">
                      {quiet ? "—" : fmt(r.pr_auc)}
                    </td>
                    <td className="px-3 py-3 tabular-nums">
                      {quiet ? "—" : fmtPct(r.top10_capture)}
                    </td>
                    <td className="px-3 py-3 tabular-nums">{fmt(r.brier)}</td>
                    <td className="px-3 py-3 tabular-nums">{quiet ? "—" : fmt(r.fss)}</td>
                    <td className="px-3 py-3 tabular-nums">{quiet ? "—" : fmt(r.rev)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
