import os
from pathlib import Path
import numpy as np
import rasterio

PATCH = 32
STRIDE = 32

# Adjust this to your normalized directory
DATA_DIR = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/normalized")

def read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
    return arr, nodata

def stack_inputs(month_a: str, month_b: str, elevation_arr: np.ndarray):
    # Inputs: [NDVI, TEMP, PRECIP] for month_a and month_b + elevation
    ndvi_a, nd_a = read_raster(DATA_DIR / f"norm_ndvi_2018_{month_a}.tif")
    temp_a, tp_a = read_raster(DATA_DIR / f"norm_tempC_2018_{month_a}.tif")
    pr_a, prn_a = read_raster(DATA_DIR / f"norm_precipMM_2018_{month_a}.tif")

    ndvi_b, nd_b = read_raster(DATA_DIR / f"norm_ndvi_2018_{month_b}.tif")
    temp_b, tp_b = read_raster(DATA_DIR / f"norm_tempC_2018_{month_b}.tif")
    pr_b, prn_b = read_raster(DATA_DIR / f"norm_precipMM_2018_{month_b}.tif")

    # Stack as H,W,C
    x = np.stack(
        [ndvi_a, temp_a, pr_a, ndvi_b, temp_b, pr_b, elevation_arr],
        axis=-1
    ).astype(np.float32)

    # Build a conservative valid mask:
    # valid if values are finite and within [0,1] for normalized layers
    # (your ROI mask should already have removed outside pixels via nodata or NaN)
    valid = np.isfinite(x).all(axis=-1)
    return x, valid

def read_label(month_target: str):
    y, y_nodata = read_raster(DATA_DIR / f"label_fire_2018_{month_target}_roiAligned.tif")
    y = y.astype(np.float32)
    valid = np.isfinite(y)
    # Force binary just in case
    y = (y >= 0.5).astype(np.uint8)
    return y, valid

def extract_patches(x_hw_c, y_hw, valid_hw, patch=32, stride=32):
    H, W, C = x_hw_c.shape

    # Trim edges so we only take full patches (simple baseline)
    Ht = (H - patch) // stride * stride + patch
    Wt = (W - patch) // stride * stride + patch

    x_hw_c = x_hw_c[:Ht, :Wt, :]
    y_hw = y_hw[:Ht, :Wt]
    valid_hw = valid_hw[:Ht, :Wt]

    X_list, Y_list = [], []

    for r in range(0, Ht - patch + 1, stride):
        for c in range(0, Wt - patch + 1, stride):
            v = valid_hw[r:r+patch, c:c+patch]
            # Keep patch only if it is fully valid
            if not v.all():
                continue
            xp = x_hw_c[r:r+patch, c:c+patch, :]
            yp = y_hw[r:r+patch, c:c+patch]
            X_list.append(xp)
            Y_list.append(yp)

    if not X_list:
        raise RuntimeError("No valid patches extracted. Check nodata handling or file paths.")

    X = np.stack(X_list, axis=0)  # N,H,W,C
    Y = np.stack(Y_list, axis=0)  # N,H,W

    # Convert to N,C,H,W for deep learning frameworks that expect channels first
    X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
    Y = Y.astype(np.uint8)

    return X, Y

def main():
    # Static elevation
    elevation, e_nodata = read_raster(DATA_DIR / "norm_elevation_2018_static_1km.tif")
    elevation = elevation.astype(np.float32)

    sequences = [
        ("01", "02", "03"),
        ("02", "03", "04"),
        ("03", "04", "05"),
    ]

    out_dir = DATA_DIR / f"patches_p{PATCH}_s{STRIDE}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for m1, m2, mt in sequences:
        x, v_x = stack_inputs(m1, m2, elevation)
        y, v_y = read_label(mt)

        valid = v_x & v_y

        X, Y = extract_patches(x, y, valid, patch=PATCH, stride=STRIDE)

        out_path = out_dir / f"Xy_in{m1}{m2}_to_{mt}_p{PATCH}_s{STRIDE}.npz"
        np.savez_compressed(out_path, X=X, y=Y)

        # Quick checks
        pos_rate = float(Y.sum()) / float(Y.size)
        print(f"Saved {out_path.name}")
        print(f"  X shape: {X.shape}  y shape: {Y.shape}")
        print(f"  Fire pixel rate in y: {pos_rate:.6f}")

if __name__ == "__main__":
    main()
