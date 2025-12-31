from pathlib import Path
import numpy as np
import rasterio

PROJECT_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/")
RAW = PROJECT_ROOT / "data_raw"
OUT = PROJECT_ROOT / "data_processed" / "masked_final"
OUT.mkdir(parents=True, exist_ok=True)

MASK_PATH = RAW / "mask" / "nepal_binary_1km_roiAligned.tif"

# List inputs explicitly to avoid picking up old files
ndvi_files = [
    RAW / "ndvi" / "ndvi_2018_01.tif",
    RAW / "ndvi" / "ndvi_2018_02.tif",
    RAW / "ndvi" / "ndvi_2018_03.tif",
    RAW / "ndvi" / "ndvi_2018_04.tif",
]
temp_files = [
    RAW / "temperature" / "tempC_2018_01.tif",
    RAW / "temperature" / "tempC_2018_02.tif",
    RAW / "temperature" / "tempC_2018_03.tif",
    RAW / "temperature" / "tempC_2018_04.tif",
]
precip_files = [
    RAW / "precipitation" / "precipMM_2018_01.tif",
    RAW / "precipitation" / "precipMM_2018_02.tif",
    RAW / "precipitation" / "precipMM_2018_03.tif",
    RAW / "precipitation" / "precipMM_2018_04.tif",
]
elev_files = [
    RAW / "elevation" / "elevation_2018_static_1km.tif",
]
fire_files = [
    RAW / "fire" / "fire_2018_03_roiAligned.tif",
    RAW / "fire" / "fire_2018_04_roiAligned.tif",
    RAW / "fire" / "fire_2018_05_roiAligned.tif",
]

NODATA = np.float32(-9999.0)

# Load Nepal binary mask
with rasterio.open(MASK_PATH) as msrc:
    mask_arr = msrc.read(1)
    nepal_valid = (mask_arr == 1)
    ref_shape = (msrc.height, msrc.width)
    ref_crs = msrc.crs
    ref_transform = msrc.transform

print("Nepal valid pixels:", int(nepal_valid.sum()), "of", nepal_valid.size)

def check_alignment(src, name):
    if (src.height, src.width) != ref_shape:
        raise ValueError(f"Shape mismatch for {name}: {(src.height, src.width)} expected {ref_shape}")
    if src.crs != ref_crs:
        raise ValueError(f"CRS mismatch for {name}: {src.crs} expected {ref_crs}")
    if src.transform != ref_transform:
        raise ValueError(f"Transform mismatch for {name}")

def mask_and_write(in_path: Path, out_path: Path):
    with rasterio.open(in_path) as src:
        check_alignment(src, in_path.name)
        arr = src.read(1).astype(np.float32)

        # Replace NaNs with NoData
        arr[np.isnan(arr)] = NODATA

        # Apply Nepal mask: outside Nepal -> NoData
        arr[~nepal_valid] = NODATA

        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=float(NODATA), compress="lzw")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)

def run_group(label, files):
    for p in files:
        out = OUT / f"masked_{p.name}"
        mask_and_write(p, out)
        print("Saved", out)

run_group("ndvi", ndvi_files)
run_group("temp", temp_files)
run_group("precip", precip_files)
run_group("elev", elev_files)
run_group("fire", fire_files)

print("Done. Final masked rasters in:", OUT)
