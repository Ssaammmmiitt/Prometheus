import rasterio
import numpy as np

p = "/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16/2020/fire16_2020_20200218.tif"
with rasterio.open(p) as ds:
    a = ds.read(1)
    print("nodata:", ds.nodata)
    print("zero_frac:", (a == 0).mean())
    print("nan_frac:", np.isnan(a).mean())




p = "/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16/2020/fire16_2020_20200218.tif"

with rasterio.open(p) as ds:
    a = ds.read(1)
    nod = ds.nodata

    valid = a != nod
    print("valid_frac:", valid.mean())

    a_valid = a[valid]
    print("zero_frac_valid:", (a_valid == 0).mean())
    print("one_frac_valid:", (a_valid == 1).mean())

    print("nodata_frac:", (a == nod).mean())
