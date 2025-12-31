from pathlib import Path
import numpy as np
import rasterio

# Paths
PROJECT_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/")

RAW = PROJECT_ROOT / "data_raw"
OUT = PROJECT_ROOT / "data_processed" / "masked"
OUT.mkdir(parents=True, exist_ok=True)

# Use one aligned fire raster as the Nepal footprint reference
FIRE_MASK_PATH = RAW / "fire"/ "alligned" / "fire_2018_04_roiAligned.tif" 

# Input rasters
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
    RAW / "fire" / "alligned" / "fire_2018_03_roiAligned.tif",
    RAW / "fire" / "alligned" / "fire_2018_04_roiAligned.tif",
    RAW / "fire" / "alligned" / "fire_2018_05_roiAligned.tif",
]

# Build Nepal mask from fire dataset mask band
with rasterio.open(FIRE_MASK_PATH) as src:
    # dataset_mask: 255 inside valid data, 0 outside mask or nodata
    nepal_mask = src.dataset_mask() > 0
    ref_profile = src.profile.copy()
    ref_shape = (src.height, src.width)

print("Nepal mask shape:", nepal_mask.shape)
print("Nepal valid pixels:", int(nepal_mask.sum()), "of", nepal_mask.size)



# Masking function
def mask_and_save(in_path: Path, out_path: Path, mask_bool: np.ndarray, ref_prof: dict):
    with rasterio.open(in_path) as src:
        if (src.height, src.width) != ref_shape:
            raise ValueError(f"Shape mismatch: {in_path.name} is {(src.height, src.width)} but expected {ref_shape}")
        if src.crs != ref_prof["crs"] or src.transform != ref_prof["transform"]:
            raise ValueError(f"Grid mismatch (CRS or transform) for {in_path.name}")

        data = src.read(1).astype(np.float32)

        # Choose a nodata value for processed rasters
        nodata_val = np.float32(-9999.0)

        # Apply Nepal mask
        data[~mask_bool] = nodata_val

        out_profile = src.profile.copy()
        out_profile.update(
            dtype="float32",
            nodata=nodata_val,
            compress="lzw"
        )

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(data, 1)


# Run masking for all rasters
all_groups = [
    ("ndvi", ndvi_files),
    ("temperature", temp_files),
    ("precipitation", precip_files),
    ("elevation", elev_files),
    ("fire", fire_files),
]

for group_name, files in all_groups:
    for p in files:
        out_name = f"masked_{p.name}"
        out_path = OUT / out_name
        mask_and_save(p, out_path, nepal_mask, ref_profile)
        print("Saved", out_path)

print("Done. Masked rasters are in", OUT)
