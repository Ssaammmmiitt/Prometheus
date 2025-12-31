from pathlib import Path
import numpy as np
import rasterio

ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus")

MASKED_DIR = ROOT / "data_processed" / "masked_final"
MASK_PATH = ROOT / "data_raw" / "mask" / "nepal_binary_1km_roiAligned.tif"

EXPECTED = [
    # NDVI
    "masked_ndvi_2018_01.tif",
    "masked_ndvi_2018_02.tif",
    "masked_ndvi_2018_03.tif",
    "masked_ndvi_2018_04.tif",
    # Temperature
    "masked_tempC_2018_01.tif",
    "masked_tempC_2018_02.tif",
    "masked_tempC_2018_03.tif",
    "masked_tempC_2018_04.tif",
    # Precipitation
    "masked_precipMM_2018_01.tif",
    "masked_precipMM_2018_02.tif",
    "masked_precipMM_2018_03.tif",
    "masked_precipMM_2018_04.tif",
    # Elevation
    "masked_elevation_2018_static_1km.tif",
    # Fire
    "masked_fire_2018_03_roiAligned.tif",
    "masked_fire_2018_04_roiAligned.tif",
    "masked_fire_2018_05_roiAligned.tif",
]

NODATA_EXPECTED = -9999.0

def fail(msg):
    print("\n❌ FAIL:", msg)
    raise SystemExit(1)

def warn(msg):
    print("⚠️  WARN:", msg)

print("Masked directory:", MASKED_DIR)
print("Mask file:", MASK_PATH)

if not MASKED_DIR.exists():
    fail("data_processed/masked_final/ does not exist. Run masking_final.py first.")

# 1) File existence check
missing = [f for f in EXPECTED if not (MASKED_DIR / f).exists()]
if missing:
    fail("Missing expected files:\n" + "\n".join(missing))
print("✅ All expected files found:", len(EXPECTED))

# 2) Load Nepal mask to know expected outside pixels
with rasterio.open(MASK_PATH) as msrc:
    mask_arr = msrc.read(1)
    nepal_valid = (mask_arr == 1)
    expected_outside = int((mask_arr == 0).sum())
    mask_shape = (msrc.height, msrc.width)
    mask_crs = msrc.crs
    mask_transform = msrc.transform

print("Nepal valid pixels:", int(nepal_valid.sum()))
print("Expected outside pixels (mask==0):", expected_outside)

# 3) Use first file as reference for alignment
ref_path = MASKED_DIR / EXPECTED[0]
with rasterio.open(ref_path) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape2 = (ref.height, ref.width)

print("\nReference raster:", ref_path.name)
print("CRS:", ref_crs, "| shape:", ref_shape2)

# Consistency with mask grid
if ref_shape2 != mask_shape:
    fail(f"Masked rasters shape {ref_shape2} does not match mask shape {mask_shape}")
if ref_crs != mask_crs:
    fail(f"Masked rasters CRS {ref_crs} does not match mask CRS {mask_crs}")
if ref_transform != mask_transform:
    fail("Masked rasters transform does not match mask transform")

print("✅ Mask grid matches masked rasters")

# 4) Per-file checks
nodata_counts = []
for fname in EXPECTED:
    p = MASKED_DIR / fname
    with rasterio.open(p) as src:
        if src.crs != ref_crs:
            fail(f"{fname}: CRS mismatch")
        if src.transform != ref_transform:
            fail(f"{fname}: transform mismatch")
        if (src.height, src.width) != ref_shape2:
            fail(f"{fname}: shape mismatch")

        nodata = src.nodata
        if nodata is None:
            fail(f"{fname}: NoData is None (should be {NODATA_EXPECTED})")
        if float(nodata) != float(NODATA_EXPECTED):
            warn(f"{fname}: NoData is {nodata} (expected {NODATA_EXPECTED})")

        arr = src.read(1)

    nan_count = int(np.isnan(arr).sum())
    if nan_count != 0:
        fail(f"{fname}: contains NaNs ({nan_count})")

    nodata_count = int((arr == NODATA_EXPECTED).sum())
    nodata_counts.append(nodata_count)

    # basic summary
    valid = arr[arr != NODATA_EXPECTED]
    if valid.size == 0:
        fail(f"{fname}: has zero valid pixels")
    vmin = float(valid.min())
    vmax = float(valid.max())

    print(f"\n{fname}")
    print("  nodata pixels:", nodata_count, "| valid min:", vmin, "| valid max:", vmax)

    # Fire-specific checks
    if "masked_fire_" in fname:
        # Fire must be 0/1 only in valid region
        uniq = np.unique(valid)
        if not np.all(np.isin(uniq, [0.0, 1.0])):
            fail(f"{fname}: fire values are not binary. Unique values: {uniq}")
        # Optional: check fire positives exist at least for Apr (usually)
        if fname.endswith("_04_roiAligned.tif"):
            positives = int((valid == 1).sum())
            print("  fire positives:", positives)
            if positives == 0:
                warn(f"{fname}: zero fire positives (could happen, but confirm visually)")

    # NDVI sanity check (scaled MODIS NDVI)
    if "masked_ndvi_" in fname:
        # NDVI scaled typically -2000..10000, outside can be masked
        if vmax > 12000 or vmin < -3000:
            warn(f"{fname}: NDVI range looks unusual (min {vmin}, max {vmax})")

    # Temperature sanity check (Celsius)
    if "masked_tempC_" in fname:
        if vmax > 60 or vmin < -40:
            warn(f"{fname}: Temperature range looks unusual (min {vmin}, max {vmax})")

    # Precip sanity check (monthly mm)
    if "masked_precipMM_" in fname:
        if vmin < 0:
            warn(f"{fname}: Precip has negative values (min {vmin})")

# 5) NoData count consistency check
nodata_counts = np.array(nodata_counts)
median_nodata = int(np.median(nodata_counts))
print("\nNoData pixel count summary")
print("  median nodata pixels:", median_nodata)
print("  expected outside pixels from Nepal mask:", expected_outside)

# We expect nodata pixels to be close to expected_outside.
# Allow some tolerance because some layers may also have internal missing data.
tolerance = int(expected_outside * 0.10)  # 10% tolerance
if abs(median_nodata - expected_outside) > tolerance:
    warn("Median NoData count differs substantially from Nepal mask outside pixels. "
         "This can happen if inputs have missing data inside Nepal, but confirm visually.")

print("\n✅ Verification complete. Masked_final looks consistent.")
