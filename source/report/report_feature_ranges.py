from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import rasterio

ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed")

VARS_YEARLY = ["ndvi16", "temp16", "precip16", "rh16", "fire16"]
STATIC_DIR = "static"
STATIC_FILES = [
    ("elevation", "elevation_static_srtm.tif"),
    ("slope", "slope_static_srtm.tif"),
]

YEARS = list(range(2018, 2026))

REPORT_CSV = ROOT / "feature_report.csv"
REPORT_JSON = ROOT / "feature_report.json"

def iter_tifs(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.tif*") if p.is_file()])

def read_valid_values(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        if nodata is None:
            valid = np.isfinite(arr)
        else:
            valid = np.isfinite(arr) & (arr != nodata)
        vals = arr[valid]
        return vals, nodata, src.dtypes[0], str(src.crs)

def safe_stats(vals: np.ndarray):
    if vals.size == 0:
        return None
    vals = vals.astype(np.float64, copy=False)
    out = {}
    out["count"] = int(vals.size)
    out["min"] = float(np.min(vals))
    out["max"] = float(np.max(vals))
    out["mean"] = float(np.mean(vals))
    out["std"] = float(np.std(vals))
    for q, name in [(1, "p01"), (5, "p05"), (50, "p50"), (95, "p95"), (99, "p99")]:
        out[name] = float(np.percentile(vals, q))
    return out

def accumulate_stats(file_list):
    """
    Compute global stats by streaming files to avoid loading everything into memory at once.
    For percentiles, we reservoir sample up to a cap.
    """
    sample_cap = 2_000_000
    sample = []
    total_count = 0
    running_sum = 0.0
    running_sumsq = 0.0
    global_min = math.inf
    global_max = -math.inf

    meta = {"nodata": None, "dtype": None, "crs": None}

    for i, f in enumerate(file_list):
        vals, nodata, dtype, crs = read_valid_values(f)
        if meta["nodata"] is None:
            meta["nodata"] = nodata
            meta["dtype"] = str(dtype)
            meta["crs"] = crs

        if vals.size == 0:
            continue

        total_count += int(vals.size)
        running_sum += float(np.sum(vals, dtype=np.float64))
        running_sumsq += float(np.sum(vals.astype(np.float64) ** 2))
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmin < global_min:
            global_min = vmin
        if vmax > global_max:
            global_max = vmax

        if sample_cap > 0:
            remaining = sample_cap - (sum(len(x) for x in sample))
            if remaining > 0:
                if vals.size <= remaining:
                    sample.append(vals.astype(np.float32, copy=False))
                else:
                    idx = np.random.choice(vals.size, size=remaining, replace=False)
                    sample.append(vals[idx].astype(np.float32, copy=False))

    if total_count == 0:
        return None, meta

    mean = running_sum / total_count
    var = (running_sumsq / total_count) - (mean ** 2)
    std = float(math.sqrt(max(var, 0.0)))

    sample_vals = np.concatenate(sample) if sample else np.array([], dtype=np.float32)
    if sample_vals.size > 0:
        p01 = float(np.percentile(sample_vals, 1))
        p05 = float(np.percentile(sample_vals, 5))
        p50 = float(np.percentile(sample_vals, 50))
        p95 = float(np.percentile(sample_vals, 95))
        p99 = float(np.percentile(sample_vals, 99))
    else:
        p01 = p05 = p50 = p95 = p99 = float("nan")

    stats = {
        "count": int(total_count),
        "min": float(global_min),
        "max": float(global_max),
        "mean": float(mean),
        "std": float(std),
        "p01": p01,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "p99": p99,
    }
    return stats, meta

def main():
    rows = []

    # Yearly variables
    for var in VARS_YEARLY:
        for year in YEARS:
            folder = ROOT / var / str(year)
            files = iter_tifs(folder)
            if not files:
                continue
            stats, meta = accumulate_stats(files)
            if stats is None:
                continue
            row = {
                "feature": var,
                "year": year,
                "n_files": len(files),
                "dtype": meta["dtype"],
                "crs": meta["crs"],
                "nodata": meta["nodata"],
                **stats
            }
            rows.append(row)

    # Static variables
    static_folder = ROOT / STATIC_DIR
    for feature_name, fname in STATIC_FILES:
        path = static_folder / fname
        if not path.exists():
            continue
        vals, nodata, dtype, crs = read_valid_values(path)
        stats = safe_stats(vals)
        if stats is None:
            continue
        row = {
            "feature": feature_name,
            "year": "static",
            "n_files": 1,
            "dtype": str(dtype),
            "crs": crs,
            "nodata": nodata,
            **stats
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rasters found under data_processed. Check folder paths.")
        return

    df = df.sort_values(["feature", "year"]).reset_index(drop=True)

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_CSV, index=False)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print("Saved:")
    print(" ", REPORT_CSV.resolve())
    print(" ", REPORT_JSON.resolve())
    print()

    # NDVI scaling check based on global max across all years in report
    ndvi_rows = df[df["feature"] == "ndvi16"]
    if not ndvi_rows.empty:
        ndvi_max = float(ndvi_rows["max"].max())
        ndvi_p99 = float(ndvi_rows["p99"].max())
        print("NDVI scaling check")
        print(" NDVI max:", ndvi_max)
        print(" NDVI p99:", ndvi_p99)

        if ndvi_max <= 1.5:
            print(" Interpretation: NDVI is already in approximately 0 to 1 scale.")
        elif ndvi_max <= 12000:
            print(" Interpretation: NDVI appears MODIS scaled. Divide NDVI by 10000 before final normalization.")
        else:
            print(" Interpretation: NDVI range is unexpected. Inspect one NDVI file manually.")
    else:
        print("No ndvi16 entries found in report to run scaling check.")

if __name__ == "__main__":
    main()
