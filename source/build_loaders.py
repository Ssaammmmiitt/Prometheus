import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from dataset_loader import FireConvLSTMDataset  # adjust if your class file name differs

DATA_ROOT = "/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized"
FIRE_ROOT = "/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16"
VARS = ["ndvi16","temp16","precip16","rh16","vpd16","elevation","slope"]

TRAIN_CSV = "/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_train_vr50.csv"
VAL_CSV   = "/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_val.csv"
TEST_CSV  = "/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_test.csv"

def make_loader(csv_path, batch_size, shuffle=False, sampler=None):
    ds = FireConvLSTMDataset(
        index_csv=csv_path,
        data_root=DATA_ROOT,
        fire_root=FIRE_ROOT,
        variables=VARS,
        patch_size=32
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    return ds, loader

def make_train_sampler(train_df):
    y = train_df["has_fire"].astype(int).to_numpy()
    counts = np.bincount(y, minlength=2)
    w0 = 1.0 / max(counts[0], 1)
    w1 = 1.0 / max(counts[1], 1)
    weights = np.where(y == 1, w1, w0).astype(np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True
    )
    return sampler, counts

if __name__ == "__main__":
    batch_size = 8

    train_ds = FireConvLSTMDataset(TRAIN_CSV, DATA_ROOT, FIRE_ROOT, VARS, patch_size=32)
    sampler, counts = make_train_sampler(train_ds.df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_ds, val_loader = make_loader(VAL_CSV, batch_size=batch_size, shuffle=False)
    test_ds, test_loader = make_loader(TEST_CSV, batch_size=batch_size, shuffle=False)

    print("Train size:", len(train_ds), "class counts [0,1]:", counts, "fire ratio:", counts[1]/counts.sum())
    print("Val size:", len(val_ds))
    print("Test size:", len(test_ds))

    X, y = next(iter(train_loader))
    print("One train batch X:", X.shape, "y:", y.shape)
    print("X range:", X.min().item(), X.max().item())
    print("y unique:", torch.unique(y))
    print("has any fire patch in batch:", (y.sum(dim=(1,2)) > 0).any().item())
