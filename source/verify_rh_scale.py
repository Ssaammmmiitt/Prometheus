from pathlib import Path
import numpy as np
import rasterio

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus")
RH_DIR = PROJECT_ROOT / "data_processed" / "rh16"
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022]

SAMPLE_CAP = 2_000_000  # keeps memory safe

# =========================
# HELPERS
# =========================
def read_valid_values(path, cap):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= (arr != nodata)

    vals = arr[valid]
    if vals.size == 0:
        return None

    if vals.size > cap:
        idx = np.random.choice(vals.size, cap, replace=False)
        vals = vals[idx]

    return vals

# =========================
# MAIN
# =========================
all_vals = []

print("Verifying RH scale using training years only\n")

for year in TRAIN_YEARS:
    year_dir = RH_DIR / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Missing RH folder: {year_dir}")

    files = sorted(year_dir.glob("*.tif*"))
    if not files:
        raise FileNotFoundError(f"No RH files in {year_dir}")

    for f in files:
        vals = read_valid_values(f, SAMPLE_CAP // len(files))
        if vals is not None:
            all_vals.append(vals)

if not all_vals:
    raise RuntimeError("No valid RH pixels found")

vals = np.concatenate(all_vals)

stats = {
    "count": int(vals.size),
    "min": float(np.min(vals)),
    "max": float(np.max(vals)),
    "p95": float(np.percentile(vals, 95)),
    "p99": float(np.percentile(vals, 99)),
}

print("RH statistics (training years)")
for k, v in stats.items():
    print(f"{k:>6}: {v}")

print("\nInterpretation:")
if stats["p99"] <= 1.2:
    print("RH is already in 0–1 range → DO NOT divide by 100")
elif stats["p99"] <= 120:
    print("RH appears to be 0–100 → divide by 100 during preprocessing")
else:
    print("Unexpected RH scale → manual inspection needed")
