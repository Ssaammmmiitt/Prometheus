from pathlib import Path
import numpy as np
import rasterio

ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus")
IN_DIR = ROOT / "data_processed" / "masked_final"
OUT_DIR = ROOT / "data_processed" / "normalized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODATA = -9999.0

ndvi_files = [IN_DIR / f"masked_ndvi_2018_0{i}.tif" for i in [1,2,3,4]]
temp_files = [IN_DIR / f"masked_tempC_2018_0{i}.tif" for i in [1,2,3,4]]
precip_files = [IN_DIR / f"masked_precipMM_2018_0{i}.tif" for i in [1,2,3,4]]
elev_file = IN_DIR / "masked_elevation_2018_static_1km.tif"

fire_files = [
    IN_DIR / "masked_fire_2018_03_roiAligned.tif",
    IN_DIR / "masked_fire_2018_04_roiAligned.tif",
    IN_DIR / "masked_fire_2018_05_roiAligned.tif",
]

def read_valid(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    valid = arr[arr != NODATA]
    return arr, valid, profile

def write_raster(path_out, arr, profile):
    prof = profile.copy()
    prof.update(dtype="float32", nodata=float(NODATA), compress="lzw")
    with rasterio.open(path_out, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)

# ---- NDVI normalization: scaled -> real -> clip -> 0..1 ----
# NDVI scaled values roughly -2000..10000, divide by 10000 gives -0.2..1.0 typical
for p in ndvi_files:
    arr, _, profile = read_valid(p)
    out = np.full_like(arr, NODATA, dtype=np.float32)

    m = (arr != NODATA)
    ndvi = arr[m] / 10000.0
    ndvi = np.clip(ndvi, -0.2, 1.0)          # optional but recommended
    ndvi01 = (ndvi + 0.2) / 1.2              # map -0.2..1.0 to 0..1

    out[m] = ndvi01
    write_raster(OUT_DIR / p.name.replace("masked_", "norm_"), out, profile)

print("NDVI normalized")

# ---- Temperature normalization: min-max across all months ----
temp_vals = []
temp_profiles = {}
temp_arrays = {}

for p in temp_files:
    arr, valid, profile = read_valid(p)
    temp_arrays[p] = arr
    temp_profiles[p] = profile
    temp_vals.append(valid)

temp_all = np.concatenate(temp_vals)
tmin, tmax = float(temp_all.min()), float(temp_all.max())
print("Temp min/max:", tmin, tmax)

for p in temp_files:
    arr = temp_arrays[p]
    profile = temp_profiles[p]
    out = np.full_like(arr, NODATA, dtype=np.float32)
    m = (arr != NODATA)
    out[m] = (arr[m] - tmin) / (tmax - tmin + 1e-8)
    write_raster(OUT_DIR / p.name.replace("masked_", "norm_"), out, profile)

print("Temperature normalized")

# ---- Precip normalization: log1p then min-max across all months ----
prec_vals = []
prec_profiles = {}
prec_arrays = {}

for p in precip_files:
    arr, valid, profile = read_valid(p)
    prec_arrays[p] = arr
    prec_profiles[p] = profile
    prec_vals.append(np.log1p(valid))   # log transform for stability

prec_all = np.concatenate(prec_vals)
pmin, pmax = float(prec_all.min()), float(prec_all.max())
print("Precip log1p min/max:", pmin, pmax)

for p in precip_files:
    arr = prec_arrays[p]
    profile = prec_profiles[p]
    out = np.full_like(arr, NODATA, dtype=np.float32)
    m = (arr != NODATA)
    logv = np.log1p(arr[m])
    out[m] = (logv - pmin) / (pmax - pmin + 1e-8)
    write_raster(OUT_DIR / p.name.replace("masked_", "norm_"), out, profile)

print("Precipitation normalized")

# ---- Elevation normalization: min-max ----
arr, valid, profile = read_valid(elev_file)
emin, emax = float(valid.min()), float(valid.max())
print("Elevation min/max:", emin, emax)

out = np.full_like(arr, NODATA, dtype=np.float32)
m = (arr != NODATA)
out[m] = (arr[m] - emin) / (emax - emin + 1e-8)
write_raster(OUT_DIR / elev_file.name.replace("masked_", "norm_"), out, profile)

print("Elevation normalized")

# ---- Fire: keep as 0/1 (float32) ----
for p in fire_files:
    arr, _, profile = read_valid(p)
    out = np.full_like(arr, NODATA, dtype=np.float32)
    m = (arr != NODATA)
    out[m] = arr[m]  # already 0/1
    write_raster(OUT_DIR / p.name.replace("masked_", "label_"), out, profile)

print("Fire labels saved")
print("Done. Normalized outputs in:", OUT_DIR)
