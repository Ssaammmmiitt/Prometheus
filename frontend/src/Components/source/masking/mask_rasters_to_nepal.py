from pathlib import Path
import numpy as np
import rasterio

PROJECT_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus")
MASK_DIR = PROJECT_ROOT / "data_raw" / "mask"

# Change this if your mask filename differs
MASK_PATH = next(MASK_DIR.glob("nepal_mask*.tif"), None)
if MASK_PATH is None:
    raise FileNotFoundError(f"No mask tif found in {MASK_DIR}")

INPUT_ROOT = PROJECT_ROOT / "data_raw"
OUTPUT_ROOT = PROJECT_ROOT / "data_processed"

VARS_YEARLY = ["ndvi16", "temp16", "precip16", "rh16", "vpd16"]
STATIC_DIR = "static"

NODATA_VALUE = -9999.0

def load_mask():
    with rasterio.open(MASK_PATH) as msrc:
        mask = msrc.read(1)
        meta = {
            "crs": msrc.crs,
            "transform": msrc.transform,
            "width": msrc.width,
            "height": msrc.height,
        }
    return mask, meta

def assert_same_grid(mask_meta, src):
    if src.crs != mask_meta["crs"]:
        raise ValueError(f"CRS mismatch: {src.crs} vs {mask_meta['crs']}")
    if src.transform != mask_meta["transform"]:
        raise ValueError("Transform mismatch (grid not aligned).")
    if src.width != mask_meta["width"] or src.height != mask_meta["height"]:
        raise ValueError("Shape mismatch (width/height not aligned).")

def mask_and_write(in_path: Path, out_path: Path, mask_arr: np.ndarray, mask_meta: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_path) as src:
        assert_same_grid(mask_meta, src)
        data = src.read(1)

        keep = mask_arr > 0
        out = data.astype(np.float32, copy=True)
        out[~keep] = NODATA_VALUE

        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, nodata=NODATA_VALUE, compress="lzw")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out.astype(np.float32), 1)

def process_yearly_var(var: str, mask_arr, mask_meta):
    var_in = INPUT_ROOT / var
    if not var_in.exists():
        return

    for year_dir in sorted([p for p in var_in.iterdir() if p.is_dir()]):
        year = year_dir.name
        out_year = OUTPUT_ROOT / var / year

        for tif in sorted(year_dir.glob("*.tif*")):
            out_path = out_year / tif.name
            mask_and_write(tif, out_path, mask_arr, mask_meta)

def process_static(mask_arr, mask_meta):
    in_dir = INPUT_ROOT / STATIC_DIR
    if not in_dir.exists():
        return

    out_dir = OUTPUT_ROOT / STATIC_DIR
    for tif in sorted(in_dir.glob("*.tif*")):
        out_path = out_dir / tif.name
        mask_and_write(tif, out_path, mask_arr, mask_meta)

def main():
    mask_arr, mask_meta = load_mask()
    print("Using mask:", MASK_PATH)

    for var in VARS_YEARLY:
        process_yearly_var(var, mask_arr, mask_meta)
        print("Masked:", var)

    process_static(mask_arr, mask_meta)
    print("Masked: static")
    print("Done. Output in:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
