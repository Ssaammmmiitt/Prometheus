#!/usr/bin/env python3
"""CLI: download FIRMS → clean → build fire_daily.zarr"""

from __future__ import annotations

import argparse
import sys

from prometheus.data.firms import build_pipeline, resolve_map_key, save_map_key


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Day-2 fire label cube from FIRMS")
    p.add_argument("--map-key", default=None, help="FIRMS MAP_KEY (or set FIRMS_MAP_KEY)")
    p.add_argument("--skip-download", action="store_true", help="Reuse cached chunks + archives")
    p.add_argument("--sleep", type=float, default=0.25, help="Seconds between API calls")
    args = p.parse_args(argv)

    if args.map_key:
        save_map_key(args.map_key)
        key = args.map_key
    else:
        try:
            key = resolve_map_key()
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 2

    print(f"Using MAP_KEY ending …{key[-4:]}")
    report = build_pipeline(map_key=key, skip_download=args.skip_download, sleep_s=args.sleep)

    print("\n=== Day 2 report ===")
    print(f"raw rows:              {report['raw_rows']:,}")
    print(f"cleaned rows (all):    {report['cleaned_rows']:,}")
    print(f"season 2016–2025:      {report['season_2016_2025_rows']:,}")
    print(f"by collection:         {report.get('by_collection_season')}")
    print(f"cube shape (T,H,W):    {report['alignment']['shape']}")
    print(f"fire pixels in cube:   {report['alignment']['total_fire_pixels']:,}")
    print(f"outside-mask fires:    {report['alignment']['outside_mask_fire_pixels']}")
    print(f"zarr:                  {report['zarr_path']}")
    print(f"year×month table:      {report['detection_table_path']}")
    if report.get("download_error"):
        print(f"download warning:      {str(report['download_error'])[:200]}")

    import pandas as pd

    table = pd.read_csv(report["detection_table_path"], index_col=0)
    print("\nDetections by year × month (cleaned, Jan–May, train years):")
    print(table.to_string())

    ok = report["alignment"]["outside_mask_fire_pixels"] == 0
    n = report["season_2016_2025_rows"]
    thr = "PASS" if n > 120_000 else "PENDING (need VIIRS download or more archives)"
    print(f"\n>120k season detections: {n:,} → {thr}")
    print(f"Outside-mask zero: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
