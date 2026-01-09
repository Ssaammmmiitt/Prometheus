import pandas as pd
from pathlib import Path

INDEX = Path("/Users/sammit/Desktop/Projects/Prometheus/reports/dataset/dataset_index_p32_s16.csv")
OUT   = INDEX.parent

df = pd.read_csv(INDEX, dtype={"t1":"string","t2":"string","t3":"string","t4":"string"})
for c in ["t1","t2","t3","t4"]:
    df[c] = df[c].str.replace(r"\.0$", "", regex=True)

train = df[df["year"].between(2018, 2023)].copy()
val   = df[df["year"] == 2024].copy()
test  = df[df["year"] == 2025].copy()

train.to_csv(OUT / "dataset_index_train.csv", index=False)
val.to_csv(OUT / "dataset_index_val.csv", index=False)
test.to_csv(OUT / "dataset_index_test.csv", index=False)

print("Saved")
print("train", len(train), "fire ratio", train["has_fire"].mean())
print("val", len(val), "fire ratio", val["has_fire"].mean())
print("test", len(test), "fire ratio", test["has_fire"].mean())
