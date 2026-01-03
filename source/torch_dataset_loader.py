from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

@dataclass(frozen=True)
class Paths:
    root_norm: Path
    root_fire: Path
    index_csv: Path
    static_elev: Path
    static_slope: Path

@dataclass(frozen=True)
class Config:
    patch_size: int = 32
    input_steps: int = 3
    channels: int = 6
    nodata_norm: float = -9999.0

TRAIN_YEARS = {2018, 2019, 2020, 2021, 2022}
VAL_YEARS = {2023}
TEST_YEARS = {2024}

FEATURE_ORDER = ["ndvi16", "temp16", "precip16", "rh16"]  # dynamic per timestep
STATIC_ORDER = ["elevation", "slope"]  # static, added as channels

def assert_exists(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {label}: {p}")

def read_patch(path: Path, row: int, col: int, size: int) -> np.ndarray:
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(col_off=col, row_off=row, width=size, height=size)
        arr = src.read(1, window=window).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr

def nan_to_num(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.where(np.isfinite(arr), arr, fill).astype(np.float32)

def quick_grid_check(reference: Path, others: list[Path]) -> None:
    with rasterio.open(reference) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

    for p in others:
        with rasterio.open(p) as src:
            if src.crs != ref_crs:
                raise ValueError(f"CRS mismatch: {p}")
            if src.transform != ref_transform:
                raise ValueError(f"Transform mismatch: {p}")
            if (src.height, src.width) != ref_shape:
                raise ValueError(f"Shape mismatch: {p}")

class FirePatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, paths: Paths, cfg: Config, do_checks: bool = True):
        self.df = df.reset_index(drop=True)
        self.paths = paths
        self.cfg = cfg

        assert_exists(paths.root_norm, "data_processed_normalized")
        assert_exists(paths.root_fire, "data_processed/fire16")
        assert_exists(paths.static_elev, "static elevation")
        assert_exists(paths.static_slope, "static slope")

        if do_checks:
            self._sanity_checks()

        self._static_cache = None  # (2, H, W) float32

    def _sanity_checks(self):
        if self.df.empty:
            raise ValueError("Dataset split has 0 rows. Check year filters and index CSV.")

        required_cols = {"year", "t1", "t2", "t3", "t4", "patch_row", "patch_col", "has_fire"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Index CSV is missing columns: {sorted(missing)}")

        for col in ["patch_row", "patch_col"]:
            if (self.df[col] < 0).any():
                raise ValueError(f"Negative {col} found in index.")

        # Grid alignment spot check using first row only
        r0 = self.df.iloc[0]
        year = int(r0["year"])
        t1 = str(r0["t1"])
        t4 = str(r0["t4"])

        ref = self.paths.root_norm / "ndvi16" / str(year) / f"ndvi16_{year}_{t1}.tif"
        temp = self.paths.root_norm / "temp16" / str(year) / f"temp16_{year}_{t1}.tif"
        precip = self.paths.root_norm / "precip16" / str(year) / f"precip16_{year}_{t1}.tif"
        rh = self.paths.root_norm / "rh16" / str(year) / f"rh16_{year}_{t1}.tif"
        fire = self.paths.root_fire / str(year) / f"fire16_{year}_{t4}.tif"

        for p, lbl in [
            (ref, "ndvi sample"),
            (temp, "temp sample"),
            (precip, "precip sample"),
            (rh, "rh sample"),
            (fire, "fire sample"),
        ]:
            assert_exists(p, lbl)

        quick_grid_check(ref, [temp, precip, rh, self.paths.static_elev, self.paths.static_slope, fire])

    def _load_static_full(self) -> np.ndarray:
        if self._static_cache is not None:
            return self._static_cache

        with rasterio.open(self.paths.static_elev) as e:
            elev = e.read(1).astype(np.float32)
            elev = np.where(elev == (e.nodata if e.nodata is not None else -9999), np.nan, elev)

        with rasterio.open(self.paths.static_slope) as s:
            slope = s.read(1).astype(np.float32)
            slope = np.where(slope == (s.nodata if s.nodata is not None else -9999), np.nan, slope)

        elev = nan_to_num(elev, 0.0)
        slope = nan_to_num(slope, 0.0)

        self._static_cache = np.stack([elev, slope], axis=0)  # (2, H, W)
        return self._static_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        year = int(row["year"])
        r = int(row["patch_row"])
        c = int(row["patch_col"])
        size = self.cfg.patch_size

        t_list = [str(row["t1"]), str(row["t2"]), str(row["t3"])]
        t4 = str(row["t4"])

        # Load dynamic channels for each timestep
        # Output tensor X: (T, C, H, W) where C=6
        X_steps = []
        for t in t_list:
            dyn = []
            dyn.append(read_patch(self.paths.root_norm / "ndvi16" / str(year) / f"ndvi16_{year}_{t}.tif", r, c, size))
            dyn.append(read_patch(self.paths.root_norm / "temp16" / str(year) / f"temp16_{year}_{t}.tif", r, c, size))
            dyn.append(read_patch(self.paths.root_norm / "precip16" / str(year) / f"precip16_{year}_{t}.tif", r, c, size))
            dyn.append(read_patch(self.paths.root_norm / "rh16" / str(year) / f"rh16_{year}_{t}.tif", r, c, size))

            dyn = [nan_to_num(a, 0.0) for a in dyn]  # replace missing with 0
            dyn = np.stack(dyn, axis=0)  # (4, H, W)

            static_full = self._load_static_full()  # (2, Hfull, Wfull)
            static_patch = static_full[:, r:r+size, c:c+size]  # (2, H, W)

            feat = np.concatenate([dyn, static_patch], axis=0)  # (6, H, W)
            X_steps.append(feat)

        X = np.stack(X_steps, axis=0).astype(np.float32)  # (T, C, H, W)

        # Load label patch
        fire_path = self.paths.root_fire / str(year) / f"fire16_{year}_{t4}.tif"
        y = read_patch(fire_path, r, c, size)
        y = (nan_to_num(y, 0.0) > 0.5).astype(np.float32)  # force 0/1

        # Torch tensors
        X_t = torch.from_numpy(X)  # (T, C, H, W)
        y_t = torch.from_numpy(y)  # (H, W)

        return X_t, y_t

def load_index(paths: Paths) -> pd.DataFrame:
    assert_exists(paths.index_csv, "dataset index CSV")
    df = pd.read_csv(paths.index_csv)
    return df

def split_by_year(df: pd.DataFrame, years: set[int]) -> pd.DataFrame:
    return df[df["year"].isin(list(years))].copy()

def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    # Oversample fire patches based on patch-level has_fire
    # This improves efficiency during early training
    # You can turn it off later and rely only on loss weighting
    has_fire = df["has_fire"].astype(int).to_numpy()
    fire_count = int(has_fire.sum())
    non_count = int(len(has_fire) - fire_count)
    if fire_count == 0:
        raise ValueError("No fire patches in this split, cannot build sampler.")

    w_fire = 1.0
    w_non = fire_count / max(non_count, 1)  # downweight non-fire
    weights = np.where(has_fire == 1, w_fire, w_non).astype(np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def compute_pos_weight_pixelwise(paths: Paths, df_train: pd.DataFrame, cfg: Config, max_samples: int = 2000) -> torch.Tensor:
    # Estimates pixel-level positive weight for BCEWithLogitsLoss
    # We sample up to max_samples patches to keep it fast
    df_s = df_train.sample(n=min(len(df_train), max_samples), random_state=42)
    pos = 0.0
    tot = 0.0
    size = cfg.patch_size
    for _, row in df_s.iterrows():
        year = int(row["year"])
        r = int(row["patch_row"])
        c = int(row["patch_col"])
        t4 = str(row["t4"])
        fire_path = paths.root_fire / str(year) / f"fire16_{year}_{t4}.tif"
        y = read_patch(fire_path, r, c, size)
        y = (nan_to_num(y, 0.0) > 0.5).astype(np.float32)
        pos += float(y.sum())
        tot += float(y.size)
    neg = tot - pos
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)

def main():
    paths = Paths(
        root_norm=Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized"),
        root_fire=Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16"),
        index_csv=Path("/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_p32_s16.csv"),
        static_elev=Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized/static/elevation_static_srtm.tif"),
        static_slope=Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized/static/slope_static_srtm.tif"),
    )
    cfg = Config(patch_size=32)

    df = load_index(paths)
    df_train = split_by_year(df, TRAIN_YEARS)
    df_val = split_by_year(df, VAL_YEARS)
    df_test = split_by_year(df, TEST_YEARS)

    print("Split sizes")
    print("Train:", len(df_train), "fire patches:", int(df_train["has_fire"].sum()))
    print("Val  :", len(df_val), "fire patches:", int(df_val["has_fire"].sum()))
    print("Test :", len(df_test), "fire patches:", int(df_test["has_fire"].sum()))

    train_ds = FirePatchDataset(df_train, paths, cfg, do_checks=True)
    val_ds = FirePatchDataset(df_val, paths, cfg, do_checks=True)

    # Option 1: loss weighting only (recommended default)
    pos_weight = compute_pos_weight_pixelwise(paths, df_train, cfg, max_samples=2000)
    print("Estimated pixel-level pos_weight for BCEWithLogitsLoss:", float(pos_weight))

    # Option 2: add sampler to oversample fire patches (efficient early training)
    sampler = make_weighted_sampler(df_train)

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

    # One batch test
    X, y = next(iter(train_loader))
    print("Batch shapes")
    print("X:", tuple(X.shape), "expected (B, T, C, H, W)")
    print("y:", tuple(y.shape), "expected (B, H, W)")
    print("X range approx:", float(X.min()), float(X.max()))
    print("y unique:", torch.unique(y))

if __name__ == "__main__":
    main()
