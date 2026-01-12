from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus").resolve()

# Use a background that actually has structure
# Prefer your Nepal binary mask if you have it
MASK_TIF = "/Users/sammit/Desktop/Projects/Prometheus/data_raw/mask/nepal_mask_1km_roiAligned.tif"
MASK_TIF = "/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized/precip16/2018/precip16_2018_20180101.tif"

# Or use any normalized raster as background
# BG_TIF = ROOT / "data_processed_normalized" / "ndvi16" / "2019" / "ndvi16_2019_20190218.tif"

CSV = ROOT /"reports"/ "dataset" / "dataset_index_test.csv"   # update if your CSV is elsewhere

PATCH = 32
N = 300

df = pd.read_csv(CSV)

# Confirm required columns
for col in ["patch_row", "patch_col"]:
    if col not in df.columns:
        raise ValueError(f"Missing column in CSV: {col}")

# Sample patches to draw
df_s = df.sample(min(N, len(df)), random_state=42).copy()

with rasterio.open(MASK_TIF) as src:
    bg = src.read(1)
    h, w = src.height, src.width
    transform = src.transform
    crs = src.crs

print("Reference raster:", MASK_TIF)
print("Raster size (H,W):", (h, w))
print("Raster CRS:", crs)

r_min, r_max = int(df_s["patch_row"].min()), int(df_s["patch_row"].max())
c_min, c_max = int(df_s["patch_col"].min()), int(df_s["patch_col"].max())
print("Sample patch_row min,max:", r_min, r_max)
print("Sample patch_col min,max:", c_min, c_max)

# Check whether patches fit inside raster bounds
outside = (
    (df_s["patch_row"] < 0)
    | (df_s["patch_col"] < 0)
    | (df_s["patch_row"] + PATCH > h)
    | (df_s["patch_col"] + PATCH > w)
)
print("Sample patches outside raster bounds:", int(outside.sum()), "/", len(df_s))

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(bg, cmap="gray", interpolation="nearest")
ax.set_title(f"Patch outlines on mask background (patch={PATCH}px), n={len(df_s)}")

# Draw rectangles in a visible color
for _, row in df_s.iterrows():
    r = int(row["patch_row"])
    c = int(row["patch_col"])
    ax.add_patch(Rectangle((c, r), PATCH, PATCH, fill=False, edgecolor="red", linewidth=1.0))

ax.set_xlim(0, w)
ax.set_ylim(h, 0)  # invert y to match raster row coordinates
ax.set_xlabel("Pixel column")
ax.set_ylabel("Pixel row")

plt.tight_layout()
plt.show()
