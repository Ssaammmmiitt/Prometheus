import re
from pathlib import Path
from collections import defaultdict
import rasterio

ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus/data_processed")

VARS = ["ndvi16", "temp16", "precip16", "rh16","vpd16"]
YEARS = [str(y) for y in range(2018, 2026)]

PAT = re.compile(r"^(?P<var>[a-z0-9]+)_(?P<year>\d{4})_(?P<date>\d{8})\.(tif|tiff)$", re.IGNORECASE)

def list_dates(var: str, year: str) -> set[str]:
    p = ROOT / var / year
    if not p.exists():
        return set()
    out = set()
    for f in p.glob("*.tif*"):
        m = PAT.match(f.name)
        if m:
            out.add(m.group("date"))
    return out

def sample_file(var: str, year: str) -> Path | None:
    p = ROOT / var / year
    if not p.exists():
        return None
    files = sorted(p.glob("*.tif*"))
    return files[0] if files else None

def grid_sig(path: Path):
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "width": src.width,
            "height": src.height,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
        }

def main():
    print("Verifying data_processed structure and alignment")
    print("Root:", ROOT.resolve())
    print()

    # 1) Date alignment check
    missing_dirs = []
    for var in VARS:
        for y in YEARS:
            if not (ROOT / var / y).exists():
                missing_dirs.append(f"{var}/{y}")

    if missing_dirs:
        print("Missing year folders under data_processed:")
        for d in missing_dirs:
            print(" ", d)
        print()

    missing = defaultdict(list)
    extra = defaultdict(list)

    for y in YEARS:
        ref = list_dates("ndvi16", y)
        if not ref:
            print(f"Year {y}: no NDVI files found, skipping date checks for this year.")
            continue

        for var in VARS:
            have = list_dates(var, y)
            if not have:
                missing[(var, y)].append("ALL_DATES_MISSING")
                continue

            m = sorted(ref - have)
            e = sorted(have - ref)

            if m:
                missing[(var, y)] = m
            if e:
                extra[(var, y)] = e

    print("Date alignment relative to NDVI")
    if not missing and not extra:
        print("OK: all variables match NDVI date stamps for all available years.")
    else:
        if missing:
            print()
            print("Missing dates:")
            for (var, y), ds in sorted(missing.items()):
                print(var, y)
                for d in ds:
                    print(" ", d)
        if extra:
            print()
            print("Extra dates not in NDVI:")
            for (var, y), ds in sorted(extra.items()):
                print(var, y)
                for d in ds:
                    print(" ", d)
    print()

    # 2) Grid consistency check per year (one sample per var)
    print("Grid consistency check (CRS, transform, shape) using one sample per variable per year")
    grid_issues = 0

    for y in YEARS:
        ref_file = sample_file("ndvi16", y)
        if ref_file is None:
            continue

        ref_sig = grid_sig(ref_file)

        for var in VARS:
            f = sample_file(var, y)
            if f is None:
                continue
            sig = grid_sig(f)

            # Compare core grid fields
            if (sig["crs"] != ref_sig["crs"] or
                sig["transform"] != ref_sig["transform"] or
                sig["width"] != ref_sig["width"] or
                sig["height"] != ref_sig["height"]):
                grid_issues += 1
                print("GRID MISMATCH:", var, y)
                print(" NDVI:", ref_file.name)
                print("  ", ref_sig["crs"], ref_sig["width"], ref_sig["height"])
                print("  ", ref_sig["transform"])
                print(" VAR :", f.name)
                print("  ", sig["crs"], sig["width"], sig["height"])
                print("  ", sig["transform"])
                print()

    if grid_issues == 0:
        print("OK: all sampled rasters match NDVI grid per year.")
    else:
        print(f"Found {grid_issues} grid mismatches. Re-export or reproject those rasters, do not proceed to training.")
    print()

    # 3) Static presence check
    static_dir = ROOT / "static"
    elev = static_dir / "elevation_static_srtm.tif"
    slope = static_dir / "slope_static_srtm.tif"

    print("Static files present")
    print(" Elevation:", elev.exists())
    print(" Slope    :", slope.exists())

    if elev.exists() and ref_file is not None:
        es = grid_sig(elev)
        if (es["crs"] != ref_sig["crs"] or es["transform"] != ref_sig["transform"]
            or es["width"] != ref_sig["width"] or es["height"] != ref_sig["height"]):
            print()
            print("WARNING: static elevation grid does not match NDVI grid.")
            print("You should re-export static layers using same scale and CRS as NDVI.")

    if slope.exists() and ref_file is not None:
        ss = grid_sig(slope)
        if (ss["crs"] != ref_sig["crs"] or ss["transform"] != ref_sig["transform"]
            or ss["width"] != ref_sig["width"] or ss["height"] != ref_sig["height"]):
            print()
            print("WARNING: static slope grid does not match NDVI grid.")
            print("You should re-export static layers using same scale and CRS as NDVI.")

if __name__ == "__main__":
    main()
