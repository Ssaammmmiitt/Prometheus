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
        patch_size=32
    ):
        """
        index_csv : path to dataset_index.csv
        data_root : data_processed_normalized
        fire_root : data_processed/fire16
        variables : list like ["ndvi16","temp16","precip16","rh16","vpd16","elevation","slope"]
        """
        self.df = pd.read_csv(index_csv)
        self.data_root = Path(data_root)
        self.fire_root = Path(fire_root)
        self.variables = variables
        self.patch = patch_size

    def __len__(self):
        return len(self.df)

    def _read_patch(self, path, r, c):
        with rasterio.open(path) as src:
            arr = src.read(1)
        return arr[r:r+self.patch, c:c+self.patch]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        year = int(row.year)
        r = int(row.patch_row)
        c = int(row.patch_col)

        # ---- INPUT SEQUENCE ----
        frames = []

        for t in [row.t1, row.t2, row.t3]:
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

        X = np.stack(frames, axis=0)  # (T, C, H, W)
        

        # ---- TARGET ----
        fire_path = self.fire_root / str(year) / f"fire16_{year}_{row.t4}.tif"
        y = self._read_patch(fire_path, r, c)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

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
print(X.min(), X.max())
print(torch.unique(y))
