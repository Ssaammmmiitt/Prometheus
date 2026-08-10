#!/usr/bin/env python3
"""Day 8: build data/cube/train_table.parquet from the feature cube + labels.

  python scripts/build_train_table.py
  python scripts/build_train_table.py --positive-cap 0     # keep every positive
  python scripts/build_train_table.py --years 2021 2022
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prometheus.config import load_settings
from prometheus.features.table import build_train_table


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument(
        "--ratio",
        type=int,
        default=settings.modeling.positive_negative_ratio,
        help="Negatives per positive",
    )
    ap.add_argument(
        "--positive-cap",
        type=int,
        default=100_000,
        help="Total positives to keep (0 = all)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    report = build_train_table(
        years=args.years,
        ratio=args.ratio,
        positive_cap=args.positive_cap or None,
        out_path=args.out,
    )

    print("\n=== Day 8 report ===")
    print(f"rows:            {report['rows']:,}")
    print(f"columns:         {report['columns']} ({report['feature_columns']} features)")
    print(f"positives:       {report['positives']:,} ({report['positive_rate']:.2%})")
    print(f"negatives ratio: 1:{report['ratio']}")
    print(f"parquet:         {report['path']} ({report['size_bytes'] / 1e6:.0f} MB)")
    print(f"norm stats:      {report['norm_stats_path']}")
    if report["missing_features"]:
        print(f"missing features: {report['missing_features']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
