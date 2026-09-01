import {
  Area,
  AreaChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function DistrictTimeseries({ rows, selectedDate }) {
  const data = (rows ?? []).map((r) => ({
    ...r,
    mean: r.mean_prob == null ? null : Number(r.mean_prob),
  }));

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="date"
            tick={{ fill: "var(--muted)", fontSize: 10, fontFamily: "JetBrains Mono" }}
            tickLine={false}
            axisLine={{ stroke: "var(--hairline)" }}
            minTickGap={24}
          />
          <YAxis
            tick={{ fill: "var(--muted)", fontSize: 10, fontFamily: "JetBrains Mono" }}
            tickLine={false}
            axisLine={false}
            width={40}
            domain={[0, "auto"]}
          />
          <Tooltip
            contentStyle={{
              background: "var(--lift)",
              border: "1px solid var(--hairline)",
              borderRadius: 0,
              fontFamily: "JetBrains Mono",
              fontSize: 12,
            }}
          />
          <Line
            type="monotone"
            dataKey="mean"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={(props) => {
              const { cx, cy, payload } = props;
              if (payload.date !== selectedDate) return null;
              return (
                <circle cx={cx} cy={cy} r={4} fill="var(--accent)" stroke="var(--surface)" />
              );
            }}
            activeDot={{ r: 3, fill: "var(--accent)" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function VerificationSparkline({ rows, selectedDate, onSelect }) {
  const data = (rows ?? []).map((r) => ({
    date: r.forecast_date,
    pr: r.valid === false || r.pr_auc == null || Number.isNaN(Number(r.pr_auc))
      ? null
      : Number(r.pr_auc),
  }));

  return (
    <div className="h-40 w-full">
      <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
        <AreaChart
          data={data}
          margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
          onClick={(state) => {
            const d = state?.activeLabel;
            if (d && onSelect) onSelect(d);
          }}
        >
          <XAxis
            dataKey="date"
            tick={{ fill: "var(--muted)", fontSize: 10, fontFamily: "JetBrains Mono" }}
            tickLine={false}
            axisLine={{ stroke: "var(--hairline)" }}
            minTickGap={32}
          />
          <YAxis
            tick={{ fill: "var(--muted)", fontSize: 10, fontFamily: "JetBrains Mono" }}
            tickLine={false}
            axisLine={false}
            width={36}
          />
          <Tooltip
            contentStyle={{
              background: "var(--lift)",
              border: "1px solid var(--hairline)",
              borderRadius: 0,
              fontFamily: "JetBrains Mono",
              fontSize: 12,
            }}
          />
          <Area
            type="monotone"
            dataKey="pr"
            stroke="var(--accent)"
            strokeWidth={2}
            fill="var(--accent)"
            fillOpacity={0.18}
            connectNulls={false}
            dot={(props) => {
              const { cx, cy, payload } = props;
              if (!cx || payload.date !== selectedDate) return null;
              return <circle cx={cx} cy={cy} r={4} fill="var(--accent)" />;
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
