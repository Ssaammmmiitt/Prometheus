import torch
from torch.utils.data import Dataset
import rasterio
import pandas as pd
import numpy as np
from pathlib import Path

class FireConvLSTMDataset(Dataset):
    def __init__(self, index_csv, data_root, fire_root, variables, patch_size=32):
        self.data_root = Path(data_root)
        self.fire_root = Path(fire_root)
        self.variables = variables
        self.patch = patch_size

        time_cols = ["t1", "t2", "t3", "t4"]
        dtype_map = {c: "string" for c in time_cols}
        self.df = pd.read_csv(index_csv, dtype=dtype_map)
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
        if nodata is not None:
            patch = patch.astype(np.float32)
            patch[patch == nodata] = -9999.0
        return patch

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        year = int(row.year)
        r = int(row.patch_row)
        c = int(row.patch_col)

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
            frame = np.stack(channels, axis=0)
            frames.append(frame)

        X_np = np.stack(frames, axis=0).astype(np.float32)
        X_np[X_np == -9999.0] = 0.0

        t4 = str(row.t4)
        fire_path = self.fire_root / str(year) / f"fire16_{year}_{t4}.tif"
        y_np = self._read_patch(fire_path, r, c).astype(np.float32)
        y_np[y_np == -9999.0] = 0.0

        X = torch.from_numpy(X_np)
        y = torch.from_numpy(y_np)
        return X, y

def scan_dataset(ds, n=300):
    fire_patch_count = 0
    total = min(n, len(ds))

    for i in range(total):
        _, y = ds[i]
        if y.sum().item() > 0:
            fire_patch_count += 1

    print("Scanned samples:", total)
    print("Fire patches in scan:", fire_patch_count)
    print("Non fire patches in scan:", total - fire_patch_count)
    print("Scan fire ratio:", fire_patch_count / total)

if __name__ == "__main__":
    DATA_ROOT = "/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized"
    FIRE_ROOT = "/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16"
    VARS = ["ndvi16","temp16","precip16","rh16","vpd16","elevation","slope"]

    SPLITS = [
        ("train_vr50", "/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_train_vr50.csv"),
        ("val",        "/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_val.csv"),
        ("test",       "/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_test.csv"),
    ]

    for name, csv_path in SPLITS:
        print("\n---", name, "---")
        ds = FireConvLSTMDataset(
            index_csv=csv_path,
            data_root=DATA_ROOT,
            fire_root=FIRE_ROOT,
            variables=VARS,
            patch_size=32
        )
        print("Dataset size:", len(ds))
        scan_dataset(ds, n=300)
