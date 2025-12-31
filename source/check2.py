import numpy as np
from pathlib import Path

p = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/normalized/patches_p32_s32")
d = np.load(p / "Xy_in0102_to_03_p32_s32.npz")
y = d["y"]
print("unique y values:", np.unique(y))
