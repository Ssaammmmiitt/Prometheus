import rasterio
from pathlib import Path

ROOT_NORM = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed_normalized")
ROOT_FIRE = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed/fire16")

REFERENCE_FILE = ROOT_NORM / "ndvi16/2018/ndvi16_2018_20180306.tif"

VARIABLES = {
    "temp16": ROOT_NORM / "temp16/2018/temp16_2018_20180306.tif",
    "precip16": ROOT_NORM / "precip16/2018/precip16_2018_20180306.tif",
    "rh16": ROOT_NORM / "rh16/2018/rh16_2018_20180306.tif",
    "fire16": ROOT_FIRE / "2018/fire16_2018_20180306.tif",
    "elevation": ROOT_NORM / "static/elevation_static_srtm.tif",
    "slope": ROOT_NORM / "static/slope_static_srtm.tif",
}

def read_meta(path):
    with rasterio.open(path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
        }

def main():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Missing reference NDVI file: {REFERENCE_FILE}")

    ref = read_meta(REFERENCE_FILE)

    print("Reference NDVI")
    print(ref)
    print("-" * 50)

    all_ok = True

    for name, path in VARIABLES.items():
        if not path.exists():
            print(f"[MISSING] {name}: {path}")
            all_ok = False
            continue

        meta = read_meta(path)

        match = (
            meta["crs"] == ref["crs"] and
            meta["transform"] == ref["transform"] and
            meta["width"] == ref["width"] and
            meta["height"] == ref["height"]
        )

        status = "OK" if match else "MISMATCH"
        print(f"{status}: {name}")
        print(meta)
        print("-" * 50)

        if not match:
            all_ok = False

    if all_ok:
        print("SUCCESS: All variables are perfectly aligned.")
    else:
        print("ERROR: Alignment issues detected. Do not proceed.")

if __name__ == "__main__":
    main()
