import numpy as np
import rasterio
from pathlib import Path

MASKED = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/masked_v2")

for fname in ["masked_ndvi_2018_01.tif", "masked_fire_2018_04_roiAligned.tif"]:
    p = MASKED / fname
    with rasterio.open(p) as src:
        arr = src.read(1)
        nodata = src.nodata
        print("\nFile:", fname)
        print("nodata:", nodata)
        print("nodata pixels:", int(np.sum(arr == nodata)), "of", arr.size)
        print("nan pixels:", int(np.sum(np.isnan(arr))))
        valid = arr[(arr != nodata) & (~np.isnan(arr))]
        print("valid min:", float(valid.min()))
        print("valid max:", float(valid.max()))
