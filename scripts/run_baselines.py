#!/usr/bin/env python3
"""Day-3: run climatology + persistence baselines (leave-one-year-out style)."""

from __future__ import annotations

import argparse
from pathlib import Path

from prometheus.config import load_settings
from prometheus.eval.cv import format_summary_table, run_baseline_loyo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Day-3 baseline evaluation (LOYO)")
    p.add_argument("--force-clim", action="store_true", help="Rebuild climatology npz")
    p.add_argument("--lookback", type=int, default=7, help="Persistence lookback days")
    args = p.parse_args(argv)

    print("Building / loading climatology (MODIS 2003–2015) …", flush=True)
    print(f"Computing persistence (lookback={args.lookback} d, 3×3 neighbourhood) …", flush=True)
    yrs = load_settings().cv.years
    print(
        f"Scoring leave-one-year-out over {yrs[0]}–{yrs[-1]} "
        f"({len(yrs)} years, Nepal mask pixels) …",
        flush=True,
    )

    per_year, summary, _ = run_baseline_loyo(
        force_clim=args.force_clim,
        lookback_days=args.lookback,
    )

    out_dir = Path(load_settings().root) / "runs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_path = out_dir / "metrics_per_year.csv"
    sum_path = out_dir / "metrics_summary.csv"
    table_path = out_dir / "metrics_table.txt"

    per_year.to_csv(per_path, index=False)
    summary.to_csv(sum_path, index=False)
    table = format_summary_table(summary)
    table_path.write_text(table + "\n", encoding="utf-8")

    print()
    print(table)
    print()
    print(f"saved: {sum_path}")
    print(f"saved: {per_path}")
    print(f"saved: {table_path}")
    print()
    print("Rule: any future model that does not beat climatology PR-AUC does not ship.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
