import pandas as pd
from pathlib import Path

# ==============================
# CONFIG
# ==============================
INPUT_ARCHIVE = Path("/Users/sammit/Desktop/Projects/Prometheus/GEE_code/Fire-Data/fire_archive_M-C61_701611.csv")
INPUT_NRT     = Path("/Users/sammit/Desktop/Projects/Prometheus/GEE_code/Fire-Data/fire_nrt_M-C61_701611.csv")

OUTPUT_CSV = Path("/Users/sammit/Desktop/Projects/Prometheus/GEE_code/Fire-Data/firms_clean_2018_2025.csv")

CONF_THRESHOLD = 50

KEEP_COLS = ["latitude", "longitude", "acq_date", "confidence"]

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def main():
    # Load both files
    df_archive = load_csv(INPUT_ARCHIVE)
    df_nrt     = load_csv(INPUT_NRT)

    print("Archive rows:", len(df_archive))
    print("NRT rows:", len(df_nrt))

    # Merge
    df = pd.concat([df_archive, df_nrt], ignore_index=True)

    print("Merged rows:", len(df))

    # Keep only required columns
    df = df[KEEP_COLS]

    # Drop missing values
    df = df.dropna(subset=KEEP_COLS)

    # Convert types
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    df = df.dropna(subset=["acq_date", "confidence"])

    # Confidence filter
    df = df[df["confidence"] >= CONF_THRESHOLD]

    # Remove exact duplicates (important)
    df = df.drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "confidence"]
    )

    # Sort chronologically
    df = df.sort_values("acq_date").reset_index(drop=True)

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Final cleaned rows:", len(df))
    print("Date range:")
    print(df["acq_date"].min(), "→", df["acq_date"].max())
    print("Saved to:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
