from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/").resolve()

IN_ROOT = PROJECT_ROOT / "data_processed"
OUT_ROOT = PROJECT_ROOT / "data_processed_normalized"

REPORT_ROOT = PROJECT_ROOT / "reports" / "normalization"

MASK_PATH = PROJECT_ROOT / "data_raw" / "mask" / "nepal_mask_1km_roiAligned.tif"

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022]

VARS_YEARLY = ["ndvi16", "temp16", "precip16", "rh16"]
STATIC_FILES = {
    "elevation": IN_ROOT / "static" / "elevation_static_srtm.tif",
    "slope": IN_ROOT / "static" / "slope_static_srtm.tif",
}

NODATA_OUT = -9999.0

SAMPLE_PER_FILE = 200_000  # for percentile sampling, keeps memory stable

# =========================
# HELPERS
# =========================
def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_year_dirs(var: str):
    p = IN_ROOT / var
    if not p.exists():
        return []
    return sorted([d for d in p.iterdir() if d.is_dir()])

def list_tifs(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.tif*") if p.is_file()])

def read_band(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        crs = src.crs
        transform = src.transform
        shape = (src.height, src.width)
    valid = np.isfinite(arr)
    if nodata is not None:
        valid = valid & (arr != nodata)
    return arr, valid, profile, nodata, crs, transform, shape

def sample_values(vals: np.ndarray, cap: int):
    if vals.size <= cap:
        return vals.astype(np.float32, copy=False)
    idx = np.random.choice(vals.size, size=cap, replace=False)
    return vals[idx].astype(np.float32, copy=False)

def stats_from_values(vals: np.ndarray):
    if vals.size == 0:
        return None
    vals = vals.astype(np.float64, copy=False)
    s = {
        "count": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p01": float(np.percentile(vals, 1)),
        "p05": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
    }
    return s

def compute_streaming_stats(files, preprocess_fn, sample_cap_total=2_000_000):
    global_min = np.inf
    global_max = -np.inf
    total_count = 0
    running_sum = 0.0
    running_sumsq = 0.0

    sample_pool = []
    sample_remaining = sample_cap_total

    meta = None

    for f in files:
        arr, valid, profile, nodata, crs, transform, shape = read_band(f)
        if meta is None:
            meta = {
                "dtype": str(profile.get("dtype", "")),
                "nodata": nodata,
                "crs": str(crs),
                "transform": tuple(transform),
                "height": shape[0],
                "width": shape[1],
            }

        arr2, v2 = preprocess_fn(arr, valid)
        if not np.any(v2):
            continue

        vals = arr2[v2]
        total_count += int(vals.size)
        running_sum += float(np.sum(vals, dtype=np.float64))
        running_sumsq += float(np.sum(vals.astype(np.float64) ** 2))

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmin < global_min:
            global_min = vmin
        if vmax > global_max:
            global_max = vmax

        if sample_remaining > 0:
            take = min(sample_remaining, SAMPLE_PER_FILE)
            samp = sample_values(vals, take)
            sample_pool.append(samp)
            sample_remaining -= int(samp.size)

    if total_count == 0:
        return None, meta

    mean = running_sum / total_count
    var = (running_sumsq / total_count) - (mean * mean)
    std = float(np.sqrt(max(var, 0.0)))

    if sample_pool:
        samp_all = np.concatenate(sample_pool)
        p01 = float(np.percentile(samp_all, 1))
        p05 = float(np.percentile(samp_all, 5))
        p50 = float(np.percentile(samp_all, 50))
        p95 = float(np.percentile(samp_all, 95))
        p99 = float(np.percentile(samp_all, 99))
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

# =========================
# PREPROCESS FUNCTIONS
# =========================
def preprocess_identity(arr, valid):
    out = arr.copy()
    out[~valid] = np.nan
    v = np.isfinite(out)
    return out, v

def preprocess_rh(arr, valid):
    out = arr.copy()
    out[~valid] = np.nan
    out = out / 100.0
    out = np.clip(out, 0.0, 1.0)
    v = np.isfinite(out)
    return out, v

def preprocess_precip(arr, valid):
    out = arr.copy()
    out[~valid] = np.nan
    out = np.log1p(np.maximum(out, 0.0))
    v = np.isfinite(out)
    return out, v

def make_preprocess_ndvi(scale_mode: str):
    """
    scale_mode options
    scaled_int means NDVI in 0..10000 with possible fill values
    unit means already in 0..1
    """
    def fn(arr, valid):
        out = arr.copy()
        out[~valid] = np.nan

        if scale_mode == "scaled_int":
            keep = np.isfinite(out) & (out >= 0) & (out <= 10000)
            out[~keep] = np.nan
            out = out / 10000.0
        else:
            keep = np.isfinite(out) & (out >= 0.0) & (out <= 1.5)
            out[~keep] = np.nan
            out = np.clip(out, 0.0, 1.0)

        v = np.isfinite(out)
        return out, v
    return fn

# NORMALIZATION
def minmax_scale(arr, valid, mn, mx):
    out = (arr - mn) / (mx - mn)
    out = np.clip(out, 0.0, 1.0)
    out[~valid] = np.nan
    return out

def write_raster(out_path: Path, arr01: np.ndarray, ref_profile: dict):
    ensure_dir(out_path.parent)
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.float32, nodata=NODATA_OUT, compress="lzw", count=1)

    out = np.where(np.isfinite(arr01), arr01.astype(np.float32, copy=False), NODATA_OUT).astype(np.float32)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)

# =========================
# MAIN
# =========================
def main():
    ensure_dir(REPORT_ROOT)
    tag = now_tag()
    log_path = REPORT_ROOT / f"normalization_log_{tag}.txt"

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("Normalization pipeline start")
    log(f"Input root: {IN_ROOT}")
    log(f"Output root: {OUT_ROOT}")
    log(f"Report root: {REPORT_ROOT}")
    log(f"Training years: {TRAIN_YEARS}")

    if not IN_ROOT.exists():
        raise FileNotFoundError(f"Missing {IN_ROOT}")

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Missing mask {MASK_PATH}")

    # Sanity check that yearly variables exist for training years
    for var in VARS_YEARLY:
        for y in TRAIN_YEARS:
            ydir = IN_ROOT / var / str(y)
            if not ydir.exists():
                raise FileNotFoundError(f"Missing training folder {ydir}")

    # NDVI scaling decision from training data only
    ndvi_train_files = []
    for y in TRAIN_YEARS:
        ndvi_train_files.extend(list_tifs(IN_ROOT / "ndvi16" / str(y)))
    if not ndvi_train_files:
        raise FileNotFoundError("No NDVI training files found")

    ndvi_raw_stats, ndvi_meta = compute_streaming_stats(ndvi_train_files, preprocess_identity)
    if ndvi_raw_stats is None:
        raise ValueError("NDVI training files have no valid pixels after nodata removal")

    ndvi_raw_max = ndvi_raw_stats["p99"]
    log("NDVI raw training stats")
    log(json.dumps(ndvi_raw_stats, indent=2))

    if ndvi_raw_max <= 1.5:
        ndvi_mode = "unit"
    else:
        ndvi_mode = "scaled_int"

    log(f"NDVI scale mode selected: {ndvi_mode}")

    preprocess_map = {
        "ndvi16": make_preprocess_ndvi(ndvi_mode),
        "temp16": preprocess_identity,
        "precip16": preprocess_precip,
        "rh16": preprocess_rh,
    }

    # Compute training stats and min max for each variable
    report_rows = []
    norm_params = {
        "training_years": TRAIN_YEARS,
        "ndvi_scale_mode": ndvi_mode,
        "features": {}
    }

    log("Computing training statistics and normalization parameters")

    # Grid reference check using one sample file
    ref_file = ndvi_train_files[0]
    _, _, _, _, ref_crs, ref_transform, ref_shape = read_band(ref_file)

    for var in VARS_YEARLY:
        train_files = []
        for y in TRAIN_YEARS:
            train_files.extend(list_tifs(IN_ROOT / var / str(y)))
        if not train_files:
            raise FileNotFoundError(f"No training files found for {var}")

        stats, meta = compute_streaming_stats(train_files, preprocess_map[var])
        if stats is None:
            raise ValueError(f"{var} has no valid pixels after preprocessing")

        # Grid check against reference
        if meta is not None:
            if meta["crs"] != str(ref_crs) or meta["transform"] != tuple(ref_transform) or (meta["height"], meta["width"]) != ref_shape:
                raise ValueError(f"Grid mismatch for {var} in training years")

        mn = stats["min"]
        mx = stats["max"]
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            raise ValueError(f"Invalid min max for {var}: {mn} {mx}")

        norm_params["features"][var] = {
            "preprocess": "ndvi_scaled_to_unit" if var == "ndvi16" else ("log1p" if var == "precip16" else ("divide_by_100" if var == "rh16" else "identity")),
            "min_train": float(mn),
            "max_train": float(mx),
            "notes": "min max computed from training years after preprocessing"
        }

        report_rows.append({
            "feature": var,
            "scope": "train_years",
            "years": ",".join(map(str, TRAIN_YEARS)),
            "n_files": len(train_files),
            "crs": meta["crs"] if meta else "",
            "height": meta["height"] if meta else "",
            "width": meta["width"] if meta else "",
            **stats
        })

        log(f"{var} train min {mn} max {mx} p95 {stats['p95']} p99 {stats['p99']}")

    # Static stats and params
    for name, path in STATIC_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing static file {path}")

        arr, valid, profile, nodata, crs, transform, shape = read_band(path)

        if str(crs) != str(ref_crs) or tuple(transform) != tuple(ref_transform) or shape != ref_shape:
            raise ValueError(f"Grid mismatch for static {name}")

        arr2, v2 = preprocess_identity(arr, valid)
        vals = arr2[v2]
        st = stats_from_values(sample_values(vals, 2_000_000)) if vals.size else None
        if st is None:
            raise ValueError(f"No valid pixels in static {name}")

        mn = float(np.nanmin(vals))
        mx = float(np.nanmax(vals))
        if mx <= mn:
            raise ValueError(f"Invalid min max for static {name}: {mn} {mx}")

        norm_params["features"][name] = {
            "preprocess": "identity",
            "min_train": mn,
            "max_train": mx,
            "notes": "min max from masked static raster"
        }

        report_rows.append({
            "feature": name,
            "scope": "static",
            "years": "static",
            "n_files": 1,
            "crs": str(crs),
            "height": shape[0],
            "width": shape[1],
            **st
        })

        log(f"{name} min {mn} max {mx}")

    # Save reports
    df_report = pd.DataFrame(report_rows)
    report_csv = REPORT_ROOT / f"train_feature_stats_{tag}.csv"
    report_json = REPORT_ROOT / f"train_feature_stats_{tag}.json"
    params_json = REPORT_ROOT / f"normalization_params_{tag}.json"

    df_report.to_csv(report_csv, index=False)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report_rows, f, indent=2)
    with open(params_json, "w", encoding="utf-8") as f:
        json.dump(norm_params, f, indent=2)

    log(f"Saved training stats CSV: {report_csv}")
    log(f"Saved training stats JSON: {report_json}")
    log(f"Saved normalization params: {params_json}")

    # Normalize all years present
    ensure_dir(OUT_ROOT)
    log("Starting normalization writes")

    for var in VARS_YEARLY:
        var_in = IN_ROOT / var
        if not var_in.exists():
            continue

        # determine all year folders that exist
        year_dirs = list_year_dirs(var)
        for ydir in year_dirs:
            year = ydir.name
            files = list_tifs(ydir)
            if not files:
                continue

            mn = norm_params["features"][var]["min_train"]
            mx = norm_params["features"][var]["max_train"]

            for p in files:
                arr, valid, profile, _, _, _, _ = read_band(p)
                arr2, v2 = preprocess_map[var](arr, valid)
                scaled = minmax_scale(arr2, v2, mn, mx)
                out_path = OUT_ROOT / var / year / p.name
                write_raster(out_path, scaled, profile)

        log(f"Wrote normalized: {var}")

    # Normalize static
    for name, path in STATIC_FILES.items():
        arr, valid, profile, _, _, _, _ = read_band(path)
        arr2, v2 = preprocess_identity(arr, valid)
        mn = norm_params["features"][name]["min_train"]
        mx = norm_params["features"][name]["max_train"]
        scaled = minmax_scale(arr2, v2, mn, mx)
        out_path = OUT_ROOT / "static" / path.name
        write_raster(out_path, scaled, profile)

    log("Wrote normalized: static")

    # Post check on a small sample to confirm 0..1 ranges
    log("Post normalization spot checks")
    spot_rows = []
    for var in ["ndvi16", "temp16", "precip16", "rh16"]:
        # pick first available year and first file
        year_dirs = list_year_dirs(var)
        if not year_dirs:
            continue
        ydir = year_dirs[0]
        files = list_tifs(OUT_ROOT / var / ydir.name)
        if not files:
            continue
        test_file = files[0]
        arr, valid, _, nodata, _, _, _ = read_band(test_file)
        vals = arr[valid]
        if vals.size == 0:
            continue
        st = stats_from_values(sample_values(vals, 500_000))
        spot_rows.append({
            "feature": var,
            "file": str(test_file.relative_to(PROJECT_ROOT)),
            "nodata": nodata,
            **st
        })
        log(f"{var} spot file {test_file.name} min {st['min']} max {st['max']} p99 {st['p99']}")

    spot_df = pd.DataFrame(spot_rows)
    spot_csv = REPORT_ROOT / f"post_normalization_spotcheck_{tag}.csv"
    spot_df.to_csv(spot_csv, index=False)
    log(f"Saved spot check CSV: {spot_csv}")

    log("Normalization pipeline complete")
    log(f"Normalized output root: {OUT_ROOT}")

if __name__ == "__main__":
    main()
