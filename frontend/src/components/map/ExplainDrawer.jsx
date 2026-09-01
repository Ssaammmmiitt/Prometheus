import { useEffect, useMemo, useState } from "react";
import { X } from "lucide-react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ApiError, getExplain } from "../../api/client";
import { RISK_WORDS } from "../../lib/plain";
import { CLASS_COLORS } from "../../lib/riskColors";
import { useForecast } from "../../state/ForecastContext";
import HorizonToggle from "./HorizonToggle";
import Button from "../ui/Button";
import Card from "../ui/Card";

function fmtPct(p, digits = 2) {
  if (p == null || Number.isNaN(Number(p))) return "—";
  return `${(Number(p) * 100).toFixed(digits)}%`;
}

function classWord(name) {
  return RISK_WORDS[name] ?? name ?? "—";
}

function badgeStyle(riskClass) {
  const fill = CLASS_COLORS[riskClass] ?? "#8a8f80";
  const darkText = riskClass == null || riskClass <= 1;
  return {
    background: fill,
    color: darkText ? "#161513" : "#ffffff",
  };
}

function CompareChart({ rows }) {
  const data = (rows ?? []).map((row) => ({
    ...row,
    pct: Number(row.probability) * 100,
    pctLabel: `${(Number(row.probability) * 100).toFixed(2)}%`,
  }));
  if (!data.length) return null;
  return (
    <div className="h-40 w-full mt-1 mb-1">
      <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 44, left: 4, bottom: 0 }}
          barCategoryGap={8}
        >
          <XAxis type="number" hide domain={[0, "auto"]} />
          <YAxis
            type="category"
            dataKey="label"
            width={108}
            tickLine={false}
            axisLine={false}
            tick={{
              fill: "var(--muted)",
              fontSize: 10,
              fontFamily: "JetBrains Mono",
            }}
          />
          <Tooltip
            cursor={{ fill: "var(--accent-12)" }}
            formatter={(value) => [`${Number(value).toFixed(2)}%`, "Chance"]}
            contentStyle={{
              background: "var(--lift)",
              border: "1px solid var(--hairline)",
              borderRadius: 0,
              fontFamily: "JetBrains Mono",
              fontSize: 12,
            }}
          />
          <Bar dataKey="pct" radius={0} maxBarSize={18}>
            {data.map((row) => (
              <Cell
                key={row.id}
                fill={row.id === "here" ? "var(--accent)" : "var(--ink)"}
                fillOpacity={row.id === "here" ? 1 : 0.35}
              />
            ))}
            <LabelList
              dataKey="pctLabel"
              position="right"
              style={{
                fill: "var(--ink)",
                fontSize: 10,
                fontFamily: "JetBrains Mono",
              }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function SnapshotGrid({ rows }) {
  if (!rows?.length) return null;
  return (
    <dl className="grid grid-cols-2 gap-x-3 gap-y-3 mt-1">
      {rows.map((row) => (
        <div key={row.key} className="min-w-0">
          <dt className="label-ui text-muted truncate">{row.label}</dt>
          <dd className="font-display font-extrabold text-xl tabular-nums leading-tight mt-0.5">
            {row.display}
            {row.unit ? (
              <span className="text-[11px] font-sans font-medium text-muted ml-1">
                {row.unit}
              </span>
            ) : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function DriverBars({ rows }) {
  if (!rows?.length) return null;
  return (
    <ul className="space-y-2.5">
      {rows.map((row) => {
        const pct = Math.round(Number(row.share) * 100);
        const up = row.direction === "up";
        return (
          <li key={row.key}>
            <div className="flex items-baseline justify-between gap-2 mb-1">
              <p className="text-sm text-ink leading-snug">{row.label}</p>
              <p className="text-xs tabular-nums text-muted shrink-0">
                {pct}%
                <span className="ml-1">{up ? "↑" : "↓"}</span>
              </p>
            </div>
            <div className="h-2 w-full bg-[var(--hairline)]">
              <span
                className="block h-full"
                style={{
                  width: `${Math.max(2, pct)}%`,
                  background: up ? "var(--accent)" : "var(--ink)",
                }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export default function ExplainDrawer({ lat, lon, district, onClose }) {
  const { date, horizon, setHorizon, query } = useForecast();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    getExplain({ lat, lon, date, horizon, top: 8 })
      .then((body) => {
        if (!cancelled) {
          setData(body);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setData(null);
          setError(err);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [lat, lon, date, horizon]);

  const offMask =
    error instanceof ApiError &&
    typeof error.detail === "string" &&
    error.detail.toLowerCase().includes("forest mask");

  const districtName = data?.district?.name ?? district?.name;
  const placeName = districtName ?? "This 1 km cell";
  const districtId = data?.district?.district_id ?? district?.district_id;
  const windowLabel = horizon === 7 ? "next 7 days" : "tomorrow";

  const ratio = useMemo(() => {
    if (data?.probability == null || data?.base_rate == null) return null;
    if (data.base_rate < 1e-9) return null;
    return data.probability / data.base_rate;
  }, [data]);

  return (
    <Card className="absolute z-950 inset-x-2 bottom-[max(0.75rem,var(--app-bottom))] max-h-[min(68dvh,calc(100dvh-var(--app-header)-1rem))] lg:inset-auto lg:top-4 lg:right-4 lg:bottom-auto lg:w-[min(400px,calc(100vw-2rem))] lg:max-h-[calc(100dvh-var(--app-header)-2rem)] w-auto overflow-y-auto overscroll-pane p-4 md:p-5 panel-enter">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <p className="label-ui text-muted">This cell</p>
          <h2 className="font-display font-bold text-lg md:text-xl leading-tight mt-1">
            {placeName}
          </h2>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="min-h-11 min-w-11 flex items-center justify-center text-muted hover:text-ink"
          aria-label="Close"
        >
          <X size={18} />
        </button>
      </div>

      <div className="mb-4">
        <HorizonToggle value={horizon} onChange={setHorizon} />
      </div>

      {loading && (
        <p className="label-ui text-muted">
          Scoring this cell and comparing it with the rest of Nepal…
        </p>
      )}

      {offMask && (
        <p className="text-sm text-muted leading-relaxed">
          This click is not in forest or grassland, so there is no score here.
          Try a greener part of the map.
        </p>
      )}

      {!loading && error && !offMask && (
        <p className="text-sm text-muted leading-relaxed">{error.message}</p>
      )}

      {data && (
        <>
          <div className="flex items-end justify-between gap-3 mb-1">
            <div>
              <p className="label-ui text-muted">Chance {windowLabel}</p>
              <p className="font-display font-extrabold text-4xl tabular-nums leading-none mt-1">
                {fmtPct(data.probability)}
              </p>
            </div>
            <span
              className="label-ui px-2 py-1 shrink-0"
              style={badgeStyle(data.risk_class)}
            >
              {classWord(data.risk_class_name)}
            </span>
          </div>

          {ratio != null && (
            <p className="text-sm text-muted leading-relaxed mt-2">
              {ratio >= 1.2
                ? `${ratio.toFixed(1)}× a typical forest day (${fmtPct(data.base_rate)})`
                : ratio <= 0.8
                  ? `${(1 / ratio).toFixed(1)}× below a typical forest day (${fmtPct(data.base_rate)})`
                  : `Close to a typical forest day (${fmtPct(data.base_rate)})`}
              {data.vs_country?.percentile != null
                ? ` · ${Math.round(data.vs_country.percentile)}th percentile nationwide`
                : ""}
            </p>
          )}

          <p className="label-ui text-muted mt-5 mb-1">How this compares</p>
          <CompareChart rows={data.compare} />

          {data.snapshot?.length > 0 && (
            <>
              <p className="label-ui text-muted mt-4 mb-2">Conditions here</p>
              <SnapshotGrid rows={data.snapshot} />
            </>
          )}

          {data.drivers?.length > 0 && (
            <>
              <p className="label-ui text-muted mt-5 mb-2">What moved this score</p>
              <DriverBars rows={data.drivers} />
              <p className="text-[11px] text-muted leading-relaxed mt-2">
                Of the themes shown, how much each moved this cell&apos;s
                score. ↑ raised the chance, ↓ lowered it. These are statistical
                weights, not a physical cause.
              </p>
            </>
          )}

          {districtId != null && (
            <Link to={`/district/${districtId}?${query}`} className="block mt-4">
              <Button variant="outline" className="w-full min-h-11">
                See all of {districtName ?? "this district"}
              </Button>
            </Link>
          )}

          <p className="text-[11px] text-muted leading-relaxed mt-4">
            {data.note}
          </p>
        </>
      )}
    </Card>
  );
}
