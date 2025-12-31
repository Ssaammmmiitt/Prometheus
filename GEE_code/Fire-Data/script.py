import pandas as pd

# 1) Change these paths
INPUT_CSV = "fireData.csv"
OUTPUT_CSV = "fire_2018_nepal_clean_conf50.csv"

# 2) Load CSV
df = pd.read_csv(INPUT_CSV)

# 3) Keep only required columns
keep_cols = ["latitude", "longitude", "acq_date", "confidence"]
missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[keep_cols].copy()

# 4) Parse date
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

# Drop rows with bad dates or missing coordinates
df = df.dropna(subset=["acq_date", "latitude", "longitude", "confidence"])

# 5) Ensure numeric confidence
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
df = df.dropna(subset=["confidence"])

# 6) Filter date range
start_date = pd.Timestamp("2018-03-01")
end_date = pd.Timestamp("2018-05-31")
df = df[(df["acq_date"] >= start_date) & (df["acq_date"] <= end_date)]

# 7) Filter confidence
CONF_THRESHOLD = 50
df = df[df["confidence"] >= CONF_THRESHOLD]

# 8) Optional: remove duplicate detections at same lat, long, date
df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "confidence"])

# 9) Save clean CSV
df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Rows after cleaning:", len(df))
print("Date range:", df["acq_date"].min(), "to", df["acq_date"].max())
print("Confidence range:", df["confidence"].min(), "to", df["confidence"].max())
