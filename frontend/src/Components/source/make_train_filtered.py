import pandas as pd
from pathlib import Path

BASE = Path("/Users/sammit/Desktop/Projects/Prometheus/reports/dataset")
train_csv = BASE / "dataset_index_train.csv"

df = pd.read_csv(train_csv, dtype={"t1":"string","t2":"string","t3":"string","t4":"string"})
for c in ["t1","t2","t3","t4"]:
    df[c] = df[c].str.replace(r"\.0$", "", regex=True)

df_f = df[df["valid_ratio"] >= 0.5].copy()
out = BASE / "dataset_index_train_vr50.csv"
df_f.to_csv(out, index=False)

print("Saved", out)
print("train_vr50", len(df_f), "fire ratio", df_f["has_fire"].mean())
