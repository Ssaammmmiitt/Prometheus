from pathlib import Path
import numpy as np
import rasterio

MASKED_DIR = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/masked_v2")

files = sorted(MASKED_DIR.glob("masked_*.tif"))

print("Verifying", len(files), "masked rasters\n")

reference_mask = None

for f in files:
    with rasterio.open(f) as src:
        data = src.read(1)
        nodata = src.nodata

        print("File:", f.name)
        print("  dtype:", data.dtype)
        print("  nodata value:", nodata)

        if nodata is None:
            print("  ❌ ERROR: NoData is None")
            continue

        nodata_pixels = np.sum(data == nodata)
        total_pixels = data.size

        print("  nodata pixels:", nodata_pixels, "/", total_pixels)

        # Build mask of valid data
        valid_mask = data != nodata

        # Save first mask as reference
        if reference_mask is None:
            reference_mask = valid_mask
        else:
            # Check mask consistency
            diff = np.sum(reference_mask != valid_mask)
            print("  mask difference vs reference:", diff)

        # Value sanity checks
        valid_data = data[valid_mask]

        if valid_data.size > 0:
            print("  min value:", float(valid_data.min()))
            print("  max value:", float(valid_data.max()))
        else:
            print("  ❌ ERROR: No valid pixels")

        print("-" * 50)

print("Verification complete.")
