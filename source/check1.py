import numpy as np
from pathlib import Path

p = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/normalized/patches_p32_s32")

for f in [
    "Xy_in0102_to_03_p32_s32.npz",
    "Xy_in0203_to_04_p32_s32.npz",
    "Xy_in0304_to_05_p32_s32.npz",
]:
    d = np.load(p / f)
    y = d["y"]
    pos_patches = (y.sum(axis=(1,2)) > 0).sum()
    print(f, "positive patches:", pos_patches, "out of", y.shape[0])
