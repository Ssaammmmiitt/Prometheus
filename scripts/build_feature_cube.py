#!/usr/bin/env python3
"""Day 6-7: build data/cube/features_daily.zarr from GEE + static layers.

  python scripts/build_feature_cube.py                 # all config years
  python scripts/build_feature_cube.py --years 2021    # one season (fast check)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prometheus.config import load_settings
from prometheus.features.cube import build_feature_cube, cube_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="*", default=None, help="Default: config years")
    ap.add_argument("--out", type=Path, default=None, help="Zarr path override")
    ap.add_argument("--keep", action="store_true", help="Fail instead of overwriting")
    args = ap.parse_args(argv)

    years = args.years or list(load_settings().years.all)
    out = args.out or cube_path()
    print(f"Building feature cube → {out}")

    report = build_feature_cube(years=years, overwrite=not args.keep, path=out)

    print("\n=== Day 6-7 report ===")
    print(f"shape (T,H,W):     {report['shape']}")
    print(f"years:             {report['years'][0]}–{report['years'][-1]}")
    print(f"dynamic variables: {len(report['variables'])}")
    print(f"static variables:  {len(report['static_variables'])}")
    print(f"forest cells:      {report['forest']['forest_cells']:,} "
          f"of {report['forest']['nepal_cells']:,} Nepal cells")
    print(f"on disk:           {report['size_bytes'] / 1e9:.2f} GB")

    print("\nNaN fraction inside forest mask:")
    for var, frac in sorted(report["nan_fraction_in_forest_mask"].items(), key=lambda kv: -kv[1]):
        flag = "  <-- over 5%" if frac > 0.05 else ""
        print(f"  {var:<18} {frac * 100:6.3f}%{flag}")

    ok = report["passes_nan_check"]
    print(f"\n<=5% NaN everywhere: {'PASS' if ok else 'FAIL'}")
    print(f"report: {report['report_path']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
