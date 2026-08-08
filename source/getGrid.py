import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window
from shapely.geometry import box
import geopandas as gpd
import numpy as np
from pathlib import Path

# -------------------------
# PATHS (CHANGE THESE)
# -------------------------
ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/")

INDEX_CSV = ROOT / "reports/dataset/dataset_index_test.csv"

REF_RASTER = (
    ROOT
    / "data_processed_normalized/ndvi16/2019/ndvi16_2019_20190218.tif"
)

OUT_PNG = ROOT / "reports/patch_overlay.png"
OUT_TIF = ROOT / "reports/patch_mask.tif"

PATCH_SIZE = 32
MAX_PATCHES_TO_PLOT = 200  # keep small for clarity

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(INDEX_CSV)
df = df.head(MAX_PATCHES_TO_PLOT)

with rasterio.open(REF_RASTER) as src:
    raster = src.read(1)
    transform = src.transform
    crs = src.crs
    height, width = raster.shape

# -------------------------
# CREATE PATCH POLYGONS
# -------------------------
polygons = []

for _, row in df.iterrows():
    r = int(row["patch_row"])
    c = int(row["patch_col"])

    # pixel coordinates -> spatial bounds
    x_min, y_max = rasterio.transform.xy(transform, r, c, offset="ul")
    x_max, y_min = rasterio.transform.xy(
        transform, r + PATCH_SIZE, c + PATCH_SIZE, offset="lr"
    )

    poly = box(x_min, y_min, x_max, y_max)
    polygons.append(poly)

gdf = gpd.GeoDataFrame(df.iloc[:len(polygons)], geometry=polygons, crs=crs)

# -------------------------
# PLOT RASTER + PATCHES
# -------------------------
plt.figure(figsize=(10, 10))
plt.imshow(raster, cmap="gray")
gdf.boundary.plot(ax=plt.gca(), color="red", linewidth=0.6)
plt.title("Patch grid overlay (32×32 pixels)")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()

print("Saved overlay image:", OUT_PNG)

# -------------------------
# OPTIONAL: SAVE PATCH MASK TIF
# -------------------------
# -------------------------
# OPTIONAL: SAVE PATCH MASK TIF (FIXED)
# -------------------------

mask = np.zeros((height, width), dtype=np.uint8)

for _, row in df.iterrows():
    r = int(row["patch_row"])
    c = int(row["patch_col"])

    r2 = min(r + PATCH_SIZE, height)
    c2 = min(c + PATCH_SIZE, width)
    mask[r:r2, c:c2] = 1

out_meta = src.meta.copy()
out_meta.update(
    dtype=rasterio.uint8,
    count=1,
    nodata=0  # valid for uint8
)

# If src.meta had a bad nodata (like -9999), the update above overrides it.
with rasterio.open(OUT_TIF, "w", **out_meta) as dst:
    dst.write(mask, 1)

print("Saved patch mask GeoTIFF:", OUT_TIF)
