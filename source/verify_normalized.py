from pathlib import Path
import numpy as np
import rasterio

ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus")
NORM_DIR = ROOT / "data_processed" / "normalized"

EXPECTED = [
    # NDVI
    "norm_ndvi_2018_01.tif",
    "norm_ndvi_2018_02.tif",
    "norm_ndvi_2018_03.tif",
    "norm_ndvi_2018_04.tif",
    # Temperature
    "norm_tempC_2018_01.tif",
    "norm_tempC_2018_02.tif",
    "norm_tempC_2018_03.tif",
    "norm_tempC_2018_04.tif",
    # Precipitation
    "norm_precipMM_2018_01.tif",
    "norm_precipMM_2018_02.tif",
    "norm_precipMM_2018_03.tif",
    "norm_precipMM_2018_04.tif",
    # Elevation
    "norm_elevation_2018_static_1km.tif",
    # Fire labels
    "label_fire_2018_03_roiAligned.tif",
    "label_fire_2018_04_roiAligned.tif",
    "label_fire_2018_05_roiAligned.tif",
]

NODATA = -9999.0

def fail(msg):
    print("\nFAIL:", msg)
    raise SystemExit(1)

missing = [f for f in EXPECTED if not (NORM_DIR / f).exists()]
if missing:
    fail("Missing files:\n" + "\n".join(missing))

print("All expected normalized files found:", len(EXPECTED))

# Use first as reference for grid consistency
with rasterio.open(NORM_DIR / EXPECTED[0]) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = (ref.height, ref.width)

print("Reference grid:", ref_shape, ref_crs)

for fname in EXPECTED:
    p = NORM_DIR / fname
    with rasterio.open(p) as src:
        if src.crs != ref_crs:
            fail(f"{fname}: CRS mismatch")
        if src.transform != ref_transform:
            fail(f"{fname}: transform mismatch")
        if (src.height, src.width) != ref_shape:
            fail(f"{fname}: shape mismatch")
        arr = src.read(1)
        nodata = src.nodata

    if nodata is None or float(nodata) != float(NODATA):
        fail(f"{fname}: NoData incorrect (got {nodata}, expected {NODATA})")

    nan_count = int(np.isnan(arr).sum())
    if nan_count != 0:
        fail(f"{fname}: contains NaNs ({nan_count})")

    nodata_count = int((arr == NODATA).sum())
    valid = arr[arr != NODATA]

    vmin = float(valid.min())
    vmax = float(valid.max())

    # Range checks
    if fname.startswith("norm_"):
        if vmin < -0.05 or vmax > 1.05:
            print("WARN:", fname, "range outside ~0..1:", vmin, vmax)
    if fname.startswith("label_fire"):
        uniq = np.unique(valid)
        if not np.all(np.isin(uniq, [0.0, 1.0])):
            fail(f"{fname}: fire labels not binary. Unique: {uniq}")

    print(fname, "| nodata:", nodata_count, "| min:", vmin, "| max:", vmax)

print("\nVerification complete.")
