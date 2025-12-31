import numpy as np
from pathlib import Path

data_dir = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/normalized/patches_p32_s32")

train_files = [
    "Xy_in0102_to_03_p32_s32.npz",
    "Xy_in0203_to_04_p32_s32.npz",
]
test_file = "Xy_in0304_to_05_p32_s32.npz"

def load_npz(fname):
    d = np.load(data_dir / fname)
    return d["X"].astype(np.float32), d["y"].astype(np.uint8)

# Load training sequences and concatenate
Xs, Ys = [], []
for f in train_files:
    X, y = load_npz(f)
    Xs.append(X)
    Ys.append(y)

X_train = np.concatenate(Xs, axis=0)
y_train = np.concatenate(Ys, axis=0)

# Load test sequence
X_test, y_test = load_npz(test_file)

# Identify positive and negative patches
patch_fire = y_train.sum(axis=(1,2))
pos_idx = np.where(patch_fire > 0)[0]
neg_idx = np.where(patch_fire == 0)[0]

print("Train patches:", len(y_train))
print("Positive patches:", len(pos_idx), "Negative patches:", len(neg_idx))

def sample_balanced_indices(seed=None):
    rng = np.random.default_rng(seed)
    n_pos = len(pos_idx)
    neg_sample = rng.choice(neg_idx, size=n_pos, replace=False if len(neg_idx) >= n_pos else True)
    idx = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(idx)
    return idx

# Example: get one epoch worth of balanced indices
epoch_idx = sample_balanced_indices(seed=42)
X_epoch = X_train[epoch_idx]
y_epoch = y_train[epoch_idx]

print("Epoch X:", X_epoch.shape, "Epoch y:", y_epoch.shape)
