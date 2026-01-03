import rasterio

ndvi = "/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized/ndvi16/2018/ndvi16_2018_20180306.tif"
fire = "/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16/2018/fire16_2018_20180306.tif"

with rasterio.open(ndvi) as a, rasterio.open(fire) as b:
    print("NDVI", a.width, a.height, a.crs)
    print("FIRE", b.width, b.height, b.crs)
    print("Transform equal:", a.transform == b.transform)
