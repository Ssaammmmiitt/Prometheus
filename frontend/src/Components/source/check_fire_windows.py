from pathlib import Path
import pandas as pd
import rasterio
from rasterio.transform import rowcol
import numpy as np
import re

FIRE_CSV = Path("/Users/sammit/Desktop/Projects/Prometheus/GEE_code/Fire-Data/firms_clean_2018_2025.csv")
MASK_TIF = Path("/Users/sammit/Desktop/Projects/Prometheus/data_raw/mask/nepal_mask_1km_roiAligned.tif")
NDVI_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/ndvi16")

WINDOW_DAYS = 16
YEARS = range(2018, 2026)
NDVI_PAT = re.compile(r"^ndvi16_(\d{4})_(\d{8})\.tif$", re.IGNORECASE)

def list_ndvi_dates(year):
    ydir = NDVI_ROOT / str(year)
    ds = []
    for f in ydir.glob("ndvi16_*.tif"):
        m = NDVI_PAT.match(f.name)
        if m:
            ds.append(pd.to_datetime(m.group(2), format="%Y%m%d"))
    return sorted(set(ds))

def main():
    df = pd.read_csv(FIRE_CSV)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "acq_date"]).copy()

    with rasterio.open(MASK_TIF) as src:
        mask = src.read(1)
        transform = src.transform
        h, w = src.height, src.width

    # Spatial filter
    keep = []
    for i, r in df.iterrows():
        lon, lat = float(r["longitude"]), float(r["latitude"])
        try:
            rr, cc = rowcol(transform, lon, lat)
        except Exception:
            continue
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] > 0:
            keep.append(i)

    df = df.loc[keep].copy()
    print("Nepal filtered fires:", len(df))
    print()

    for y in YEARS:
        dates = list_ndvi_dates(y)
        if not dates:
            continue

        print("Year", y)
        for d0 in dates:
            d1 = d0 + pd.Timedelta(days=WINDOW_DAYS)
            win = df[(df["acq_date"] >= d0) & (df["acq_date"] < d1)]
            print(d0.strftime("%Y%m%d"), "points", len(win))
        print()

if __name__ == "__main__":
    main()
