import { useEffect, useMemo, useState } from "react";
import { X } from "lucide-react";
import { Link } from "react-router-dom";

import { ApiError, getExplain } from "../../api/client";
import { explainHeadline, prepareExplain } from "../../lib/plain";
import Button from "../ui/Button";
import Card from "../ui/Card";

function FactorBars({ title, rows, up }) {
  if (!rows.length) return null;
  return (
    <div className="mb-4 last:mb-0">
      <p className="label-ui text-muted mb-2">{title}</p>
      <ul className="space-y-2">
        {rows.map((row, i) => (
          <li key={row.key}>
            <p className="text-sm text-ink leading-snug mb-1">{row.label}</p>
            <div className="h-2 w-full bg-[var(--hairline)]">
              <span
                className="block h-full"
                style={{
                  width: `${row.pct}%`,
                  background:
                    i === 0
                      ? up
                        ? "var(--accent)"
                        : "var(--ink)"
                      : "var(--muted)",
                }}
              />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ExplainDrawer({ lat, lon, date, horizon, query, district, onClose }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
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

  const factors = useMemo(() => prepareExplain(data?.top, 4), [data]);
  const hotter = factors.filter((row) => row.up);
  const quieter = factors.filter((row) => !row.up);

  const offMask =
    error instanceof ApiError &&
    typeof error.detail === "string" &&
    error.detail.toLowerCase().includes("forest mask");

  return (
    <Card className="absolute top-[11.5rem] md:top-20 right-2 md:right-4 z-900 w-[min(340px,calc(100vw-1rem))] max-h-[min(52vh,calc(100vh-16rem))] md:max-h-[calc(100vh-10rem)] overflow-y-auto p-4 md:p-5 panel-enter">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <p className="label-ui text-muted">Why here?</p>
          <h2 className="font-display font-bold text-lg md:text-xl leading-tight mt-1">
            {district?.name ?? "This patch of forest"}
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

      {district && (
        <Link to={`/district/${district.district_id}?${query}`} className="block mb-4">
          <Button variant="outline" className="w-full min-h-11">
            Open {district.name}
          </Button>
        </Link>
      )}

      {loading && <p className="label-ui text-muted">Looking this up…</p>}

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
          <p className="text-sm text-ink leading-relaxed mb-4">
            {explainHeadline(factors[0])}
          </p>
          <FactorBars title="Pushing danger up" rows={hotter} up />
          <FactorBars title="Keeping it quieter" rows={quieter} up={false} />
        </>
      )}
    </Card>
  );
}
