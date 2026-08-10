#!/usr/bin/env python3
"""Inventory data/raw/gee + data/static coverage and optional (1) dedupe.

  python scripts/audit_gee_raw.py
  python scripts/audit_gee_raw.py --dedupe   # remove Drive-style name (1) dups
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

YEARS = range(2016, 2027)
MONTHS = [1, 2, 3, 4, 5]
GEE = ROOT / "data" / "raw" / "gee"
STATIC = ROOT / "data" / "static"


def _tifs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))


def _date_key(path: Path) -> str | None:
    stem = re.sub(r"\s*\(\d+\)\s*$", "", path.stem)
    m = re.search(r"(\d{8})", stem)
    return m.group(1) if m else None


def dedupe_modis(folder: Path, prefix: str) -> tuple[int, int]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in _tifs(folder):
        k = _date_key(p)
        if k:
            groups[k].append(p)
    removed = 0
    for key, flist in groups.items():
        best = max(flist, key=lambda p: (p.stat().st_size, p.stat().st_mtime))
        target = folder / f"{prefix}_{key}.tif"
        for p in flist:
            if p.resolve() != best.resolve():
                p.unlink()
                removed += 1
        if best.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            best.rename(target)
    return len(groups), removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dedupe", action="store_true", help="Remove smaller Drive (1) duplicates")
    args = ap.parse_args()

    if args.dedupe:
        for folder, prefix in (("ndvi", "ndvi"), ("lst", "lst")):
            n, rem = dedupe_modis(GEE / folder, prefix)
            print(f"dedupe {folder}: kept {n}, removed {rem}")

    exp_era5 = {f"era5_{y}_{m:02d}.tif" for y in YEARS for m in MONTHS}
    era5 = {p.name for p in _tifs(GEE / "era5")}
    print(f"ERA5: {len(era5)}/{len(exp_era5)} missing={sorted(exp_era5 - era5)}")

    for folder, prefix, floor in (("ndvi", "ndvi", 10), ("lst", "lst", 19)):
        keys = []
        for p in _tifs(GEE / folder):
            k = _date_key(p)
            if k:
                keys.append(k)
        by_y = Counter(int(k[:4]) for k in keys if int(k[4:6]) in MONTHS)
        print(f"{folder}: {len(keys)} files by year={dict(sorted(by_y.items()))}")
        low = [y for y in YEARS if by_y.get(y, 0) < floor]
        if low:
            print(f"  WARNING years < {floor}: {low}")

    for name in ("elev_slope_aspect.tif", "twi.tif", "worldcover_frac.tif"):
        print(f"gee/static/{name}: {'OK' if (GEE / 'static' / name).is_file() else 'MISSING'}")

    from prometheus.grid import assert_aligned

    for name in (
        "nepal_mask_1km_roiAligned.tif",
        "elevation_static_srtm.tif",
        "slope_static_srtm.tif",
        "dist_road.tif",
        "dist_settlement.tif",
        "physio_regions.tif",
    ):
        p = STATIC / name
        if not p.is_file():
            print(f"static/{name}: MISSING")
            continue
        try:
            assert_aligned(p)
            print(f"static/{name}: OK aligned")
        except Exception as e:
            print(f"static/{name}: FAIL {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
