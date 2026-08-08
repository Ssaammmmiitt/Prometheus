#!/usr/bin/env python3
"""
Min Max scanner for already masked rasters.

Reads all .tif files under data_processed, groups by variable
(ndvi16, temp16, precip16, rh16, vpd16, fire16, static)
and computes per variable min and max across all files.

Assumptions
- Rasters are already masked, so outside Nepal is NoData or NaN
- We ignore NoData and non finite values when computing min max
- For "static", this script reports BOTH slope and elevation separately
  if filenames contain those words, otherwise it reports a combined static min max

Outputs
- reports/dataset/minmax_by_variable.csv
- prints a concise summary table to terminal

Requires
pip install rasterio numpy pandas tqdm
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


# =========================
# CONFIG
# =========================
ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus").resolve()
DATA_PROCESSED = ROOT / "data_processed"

REPORT_DIR = ROOT / "reports" / "dataset"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = REPORT_DIR / "minmax_by_variable.csv"

RASTER_EXTS = {".tif", ".tiff"}

# your expected groups
KNOWN_VARS = {"ndvi16", "temp16", "precip16", "rh16", "vpd16", "fire16", "static"}
STATIC_SUBVARS = {"slope", "elevation"}  # detect these within static


# =========================
# HELPERS
# =========================
def infer_var(path: Path) -> str:
    s = path.as_posix().lower()
    for v in KNOWN_VARS:
        if f"/{v}/" in s or re.search(rf"\b{re.escape(v)}\b", path.stem.lower()):
            return v
        if v in path.stem.lower():
            return v
    return "unknown"


def infer_static_subvar(path: Path) -> str:
    name = path.stem.lower()
    for sv in STATIC_SUBVARS:
        if sv in name:
            return sv
    return "static_combined"


def valid_values(arr: np.ndarray, nodata) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)

    # handle nodata value if defined
    if nodata is not None and not np.isnan(nodata):
        a = np.where(a == nodata, np.nan, a)

    # keep finite only
    return a[np.isfinite(a)]


# =========================
# MAIN
# =========================
def main() -> None:
    if not DATA_PROCESSED.exists():
        raise FileNotFoundError(f"Missing folder: {DATA_PROCESSED}")

    rasters = sorted([p for p in DATA_PROCESSED.rglob("*") if p.suffix.lower() in RASTER_EXTS])
    if not rasters:
        raise RuntimeError(f"No rasters found under: {DATA_PROCESSED}")

    # stats dict: key -> {min, max, files, pixels}
    stats = {}
    def init_key(k):
        if k not in stats:
            stats[k] = {"min": math.inf, "max": -math.inf, "files": 0, "finite_pixels": 0}

    for fp in tqdm(rasters, desc="Scanning rasters"):
        var = infer_var(fp)
        if var == "unknown":
            continue

        # static sub split
        if var == "static":
            sub = infer_static_subvar(fp)
            key = f"static_{sub}"
        else:
            key = var

        init_key(key)

        with rasterio.open(fp) as src:
            arr = src.read(1)
            nodata = src.nodata

        v = valid_values(arr, nodata)
        stats[key]["files"] += 1

        if v.size == 0:
            continue

        stats[key]["finite_pixels"] += int(v.size)

        mn = float(v.min())
        mx = float(v.max())

        if mn < stats[key]["min"]:
            stats[key]["min"] = mn
        if mx > stats[key]["max"]:
            stats[key]["max"] = mx

    # build dataframe
    rows = []
    for k, d in sorted(stats.items(), key=lambda x: x[0]):
        rows.append(
            {
                "variable": k,
                "files": d["files"],
                "finite_pixels": d["finite_pixels"],
                "min": (d["min"] if d["min"] != math.inf else float("nan")),
                "max": (d["max"] if d["max"] != -math.inf else float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\nSaved:", OUT_CSV)
    print("\nMin Max summary:")
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No matching rasters found for expected variables.")


if __name__ == "__main__":
    main()
