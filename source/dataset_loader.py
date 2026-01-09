import torch
from torch.utils.data import Dataset
import rasterio
import pandas as pd
import numpy as np
from pathlib import Path

class FireConvLSTMDataset(Dataset):
    def __init__(
        self,
        index_csv,
        data_root,
        fire_root,
        variables,
        patch_size=32,
        valid_ratio_threshold=0.5
    ):
        self.data_root = Path(data_root)
        self.fire_root = Path(fire_root)
        self.variables = variables
        self.patch = patch_size
        self.valid_ratio_threshold = valid_ratio_threshold

        # Force timestep columns to string so you never get ".0" in filenames
        # Adjust these names if your CSV uses different column names
        time_cols = ["t1", "t2", "t3", "t4"]
        dtype_map = {c: "string" for c in time_cols}
        self.df = pd.read_csv(index_csv, dtype=dtype_map)

        # Safety cleanup if the CSV already contains values like "20180117.0"
        for c in time_cols:
            self.df[c] = self.df[c].str.replace(r"\.0$", "", regex=True)

    def __len__(self):
        return len(self.df)

    def _read_patch(self, path, r, c):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing raster: {path}")

        with rasterio.open(path) as src:
            arr = src.read(1)
            nodata = src.nodata

        patch = arr[r:r + self.patch, c:c + self.patch]

        # Convert NoData to a consistent sentinel if nodata exists
        if nodata is not None:
            patch = patch.astype(np.float32)
            patch[patch == nodata] = -9999.0

        return patch

    @staticmethod
    def _tok(x) -> str:
        # Extra safety if something slips through
        s = str(x)
        if s.endswith(".0"):
            s = s[:-2]
        return s

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        year = int(row.year)
        r = int(row.patch_row)
        c = int(row.patch_col)

        # INPUT SEQUENCE
        frames = []
        t_list = [self._tok(row.t1), self._tok(row.t2), self._tok(row.t3)]

        for t in t_list:
            channels = []
            for var in self.variables:
                if var in ["elevation", "slope"]:
                    path = self.data_root / "static" / f"{var}_static_srtm.tif"
                else:
                    path = self.data_root / var / str(year) / f"{var}_{year}_{t}.tif"

                patch = self._read_patch(path, r, c)
                channels.append(patch)

            frame = np.stack(channels, axis=0)  # (C, H, W)
            frames.append(frame)

        X_np = np.stack(frames, axis=0)  # (T, C, H, W)

        # Valid ratio check using NDVI at t1, treating -9999 as invalid
        ndvi_t1 = X_np[0, 0]
        valid_ratio = float(np.mean(ndvi_t1 != -9999.0))
        if valid_ratio < self.valid_ratio_threshold:
            print("Low valid NDVI ratio:", valid_ratio, "year:", year, "t1:", row.t1, "r:", r, "c:", c)

        # Replace -9999 with 0 so tensors stay within [0, 1] after normalization
        X_np = X_np.astype(np.float32)
        X_np[X_np == -9999.0] = 0.0

        # TARGET
        t4 = self._tok(row.t4)
        fire_path = self.fire_root / str(year) / f"fire16_{year}_{t4}.tif"
        y_np = self._read_patch(fire_path, r, c).astype(np.float32)

        X = torch.from_numpy(X_np)  # (T, C, H, W)
        y = torch.from_numpy(y_np)  # (H, W)

        # If fire labels ever contain NoData, set it to 0
        y[y == -9999.0] = 0.0

        # If your inputs are already normalized to [0,1], this should hold
        if not (torch.all(X >= 0.0) and torch.all(X <= 1.0)):
            xmin = float(X.min())
            xmax = float(X.max())
            raise AssertionError(f"Input out of [0,1] range. min={xmin}, max={xmax}")

        return X, y


ds = FireConvLSTMDataset(
    index_csv="/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_p32_s16.csv",
    data_root="/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized",
    fire_root="/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16",
    variables=["ndvi16","temp16","precip16","rh16","vpd16","elevation","slope"]
)

X, y = ds[0]
print(X.shape)  # (3, 7, 32, 32)
print(y.shape)  # (32, 32)
print(X.min().item(), X.max().item())
print(torch.unique(y))


fire_found = 0
nonfire_found = 0

for i in range(200):
    X, y = ds[i]
    s = float(y.sum().item())
    if s > 0:
        fire_found += 1
    else:
        nonfire_found += 1

print("First 200 samples")
print("fire patches:", fire_found)
print("non fire patches:", nonfire_found)