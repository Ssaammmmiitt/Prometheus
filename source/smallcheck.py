import rasterio
import numpy as np

p = "/Users/sammit/Desktop/Projects/Prometheus/data_processed/masked_final/masked_ndvi_2018_01.tif"

with rasterio.open(p) as src:
    arr = src.read(1)
    print("shape:", arr.shape)
    print("unique values:", np.unique(arr))
    print("count of 1s:", int((arr == 1).sum()))
    print("count of 0s:", int((arr == 0).sum()))
