from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/")
PROCESSED_ROOT = PROJECT_ROOT / "data_processed"

# Use your cleaned merged CSV
FIRE_CSV = PROJECT_ROOT / "GEE_code" / "Fire-Data" / "firms_clean_2018_2025.csv"

# Nepal mask (used to exclude outside pixels reliably)
MASK_PATH = PROJECT_ROOT / "data_raw" / "mask" / "nepal_mask_1km_roiAligned.tif"

# NDVI folder (used as truth for available dates)
NDVI_ROOT = PROCESSED_ROOT / "ndvi16"

# Output folder
OUT_ROOT = PROCESSED_ROOT / "fire16"

# Years to build
YEARS = list(range(2018, 2026))

WINDOW_DAYS = 16

# Fire CSV columns
LAT_COL = "latitude"
LON_COL = "longitude"
DATE_COL = "acq_date"

# Filename pattern: ndvi16_YYYY_YYYYMMDD.tif
NDVI_PAT = re.compile(r"^ndvi16_(\d{4})_(\d{8})\.tif$", re.IGNORECASE)

# =========================
# HELPERS
# =========================
def load_reference_grid():
    """
    Open mask as the authoritative grid because you already verified alignment in data_processed.
    """
    with rasterio.open(MASK_PATH) as src:
        mask = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        nodata = src.nodata
    return mask, profile, transform, crs, height, width, nodata

def list_ndvi_dates(year: int) -> list[pd.Timestamp]:
    year_dir = NDVI_ROOT / str(year)
    if not year_dir.exists():
        return []

    dates = []
    for f in year_dir.glob("ndvi16_*.tif"):
        m = NDVI_PAT.match(f.name)
        if not m:
            continue
        d = pd.to_datetime(m.group(2), format="%Y%m%d", errors="coerce")
        if pd.notna(d):
            dates.append(d)

    dates = sorted(set(dates))
    return dates

def rasterize_points_binary(lons: np.ndarray, lats: np.ndarray, transform, height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint8)
    for lon, lat in zip(lons, lats):
        try:
            r, c = rowcol(transform, lon, lat)
        except Exception:
            continue
        if 0 <= r < height and 0 <= c < width:
            out[r, c] = 1
    return out

def main():
    if not FIRE_CSV.exists():
        raise FileNotFoundError(FIRE_CSV)
    if not MASK_PATH.exists():
        raise FileNotFoundError(MASK_PATH)

    mask_arr, mask_profile, transform, crs, height, width, _ = load_reference_grid()

    # Build output profile for label rasters
    out_profile = mask_profile.copy()
    out_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )

    # Load and clean fire CSV
    df = pd.read_csv(FIRE_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[LAT_COL, LON_COL, DATE_COL])

    # Ensure numeric
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df = df.dropna(subset=[LAT_COL, LON_COL])

    # Spatial filter to Nepal using the raster mask grid
    # Keep point if it falls on a pixel where mask > 0
    keep_rows = []
    for idx, row in df.iterrows():
        lon = float(row[LON_COL])
        lat = float(row[LAT_COL])
        try:
            r, c = rowcol(transform, lon, lat)
        except Exception:
            continue
        if 0 <= r < height and 0 <= c < width and mask_arr[r, c] > 0:
            keep_rows.append(idx)

    df = df.loc[keep_rows].copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    print("Fires after Nepal-only spatial filter:", len(df))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        dates = list_ndvi_dates(year)
        if not dates:
            print("No NDVI dates for year", year, "skipping.")
            continue

        year_out = OUT_ROOT / str(year)
        year_out.mkdir(parents=True, exist_ok=True)

        windows_with_fire = 0

        for d0 in dates:
            d1 = d0 + pd.Timedelta(days=WINDOW_DAYS)

            win = df[(df[DATE_COL] >= d0) & (df[DATE_COL] < d1)]
            if len(win) == 0:
                label = np.zeros((height, width), dtype=np.uint8)
            else:
                label = rasterize_points_binary(
                    win[LON_COL].to_numpy(),
                    win[LAT_COL].to_numpy(),
                    transform,
                    height,
                    width
                )
                # Apply Nepal mask defensively (ensure outside is 0)
                label = (label * (mask_arr > 0).astype(np.uint8))

            if label.sum() > 0:
                windows_with_fire += 1

            out_name = f"fire16_{year}_{d0.strftime('%Y%m%d')}.tif"
            out_path = year_out / out_name

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(label, 1)

        print(f"Year {year}: windows={len(dates)} windows_with_fire={windows_with_fire}")

    print("Done. Labels written to:", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
