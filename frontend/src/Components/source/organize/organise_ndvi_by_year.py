import re
import shutil
from pathlib import Path


# CONFIG
NDVI_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/data_raw/ndvi_16")
PATTERN = re.compile(r"ndvi16_(\d{4})_\d{8}\.tif$", re.IGNORECASE)

def main():
    if not NDVI_ROOT.exists():
        print(f"Folder not found: {NDVI_ROOT}")
        return

    files = [f for f in NDVI_ROOT.iterdir() if f.is_file() and f.suffix.lower() == ".tif"]

    if not files:
        print("No NDVI files found.")
        return

    moved = 0
    skipped = 0

    for f in files:
        m = PATTERN.match(f.name)
        if not m:
            print(f"Skipping (name does not match): {f.name}")
            skipped += 1
            continue

        year = m.group(1)
        year_dir = NDVI_ROOT / year
        year_dir.mkdir(exist_ok=True)

        dest = year_dir / f.name

        if dest.exists():
            print(f"Already exists, skipping: {dest}")
            skipped += 1
            continue

        shutil.copy2(f, dest)
        moved += 1

    print("Done.")
    print(f"Copied files: {moved}")
    print(f"Skipped files: {skipped}")

if __name__ == "__main__":
    main()
