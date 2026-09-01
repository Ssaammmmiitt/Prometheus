#!/usr/bin/env python3
"""Day 13: produce risk COGs + district GeoJSON for one date or a season range.

  python scripts/forecast.py --date 2025-04-12
  python scripts/forecast.py --backfill 2024 2025 2026
  python scripts/forecast.py --verify 2024-01-01 2026-05-30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prometheus.infer import verify as ver
from prometheus.infer.forecast import ForecastPipeline, is_in_season
from prometheus.models.predict import _as_date


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument(
        "--backfill", type=int, nargs="*", default=None,
        help="Years to backfill (Jan–May), e.g. 2024 2025",
    )
    ap.add_argument(
        "--verify", type=str, nargs=2, metavar=("START", "END"), default=None,
        help="Score h1 COGs against next-day fires over a date range",
    )
    ap.add_argument("--force", action="store_true", help="Rewrite existing outputs")
    ap.add_argument("--bundle", default="latest")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if not any([args.date, args.backfill is not None, args.verify]):
        ap.error("provide --date and/or --backfill and/or --verify")

    pipe = ForecastPipeline(bundle=args.bundle, out_dir=args.out_dir)

    if args.date:
        day = _as_date(args.date)
        if not is_in_season(day):
            print(f"{day} is outside Jan–May — no forecast produced")
            return 1
        result = pipe.forecast(day, force=args.force)
        verb = "skipped (already on disk)" if result.skipped else f"wrote in {result.seconds:.1f}s"
        print(f"{day}: {verb}")
        for key, path in result.paths.items():
            print(f"  {key}: {path}")

    if args.backfill is not None:
        years = args.backfill or [2024, 2025, 2026]
        print(f"backfill years {years} → {pipe.out_dir}")
        results = pipe.backfill(years, force=args.force)
        written = sum(1 for r in results if not r.skipped)
        print(f"done: {written} written, {len(results) - written} skipped")

    if args.verify:
        frame = ver.verify_range(args.verify[0], args.verify[1], root=pipe.out_dir)
        out = ver.verification_path(pipe.out_dir)
        if frame.empty:
            print("no forecasts to verify in that range")
            return 1
        valid = frame[frame["valid"] == True]  # noqa: E712
        print(f"verification → {out}  ({len(frame)} days, {len(valid)} with fires)")
        if not valid.empty:
            print(f"  mean PR-AUC {valid.pr_auc.mean():.4f}  "
                  f"mean top10 {valid.top10_capture.mean():.3f}  "
                  f"mean Brier {valid.brier.mean():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
