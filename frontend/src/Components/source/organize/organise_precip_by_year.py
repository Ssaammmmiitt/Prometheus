import re
import shutil
from pathlib import Path

PRECIP_ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/data_raw/precip16")
PATTERN = re.compile(r"precip16_(\d{4})_\d{8}\.tif$", re.IGNORECASE)

def main():
    if not PRECIP_ROOT.exists():
        print(f"Folder not found: {PRECIP_ROOT}")
        return

    files = [f for f in PRECIP_ROOT.iterdir() if f.is_file() and f.suffix.lower() == ".tif"]

    moved, skipped = 0, 0

    for f in files:
        m = PATTERN.match(f.name)
        if not m:
            print(f"Skipping (name mismatch): {f.name}")
            skipped += 1
            continue

        year = m.group(1)
        year_dir = PRECIP_ROOT / year
        year_dir.mkdir(exist_ok=True)

        dest = year_dir / f.name
        if dest.exists():
            skipped += 1
            continue

        shutil.copy2(f, dest)
        moved += 1

    print(f"Copied: {moved}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
