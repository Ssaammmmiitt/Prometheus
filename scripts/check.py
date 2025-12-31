import rasterio


pathNdvi = "/Users/sammit/Desktop/Projects/Prometheus/data_raw/ndvi/ndvi_2018_01.tif"

with rasterio.open(pathNdvi) as src:
    print("NDVI File Info:")
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Transform:", src.transform)
    print("Resolution:", src.res)
    print("Count (bands):", src.count)
    print("Dtype:", src.dtypes)
    print("NoData:", src.nodata)
    print("\n \n")


pathPrecipitation = "/Users/sammit/Desktop/Projects/Prometheus/data_raw/precipitation/precipMM_2018_01.tif"

with rasterio.open(pathPrecipitation) as src:
    print("Precipitation File Info:")
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Transform:", src.transform)
    print("Resolution:", src.res)
    print("Count (bands):", src.count)
    print("Dtype:", src.dtypes)
    print("NoData:", src.nodata)
    print("\n \n")


pathTemperature = "/Users/sammit/Desktop/Projects/Prometheus/data_raw/temperature/tempC_2018_01.tif"
with rasterio.open(pathTemperature) as src:
    print("Temperature File Info:")
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Transform:", src.transform)
    print("Resolution:", src.res)
    print("Count (bands):", src.count)
    print("Dtype:", src.dtypes)
    print("NoData:", src.nodata)
    print("\n \n")


pathElevation = "/Users/sammit/Desktop/Projects/Prometheus/data_raw/elevation/elevation_2018_static_1km.tif"
with rasterio.open(pathElevation) as src:
    print("Elevation File Info:")
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Transform:", src.transform)
    print("Resolution:", src.res)
    print("Count (bands):", src.count)
    print("Dtype:", src.dtypes)
    print("NoData:", src.nodata)
    print("\n \n")


pathFireData ="/Users/sammit/Desktop/Projects/Prometheus/data_raw/fire/alligned/fire_2018_04_roiAligned.tif"
with rasterio.open(pathFireData) as src:
    print("Fire Data File Info:")
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Transform:", src.transform)
    print("Resolution:", src.res)
    print("Count (bands):", src.count)
    print("Dtype:", src.dtypes)
    print("NoData:", src.nodata)
    print("\n \n")

pathFireDataOld ="/Users/sammit/Desktop/Projects/Prometheus/data_raw/fire/fire_2018_04.tif"
with rasterio.open(pathFireDataOld) as src:
    print("Old Fire Data File Info:")
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Transform:", src.transform)
    print("Resolution:", src.res)
    print("Count (bands):", src.count)
    print("Dtype:", src.dtypes)
    print("NoData:", src.nodata)
    print("\n \n")