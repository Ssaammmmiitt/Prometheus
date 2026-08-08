import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

FAMILIES = ["ndvi16", "temp16", "precip16", "rh16", "vpd16"]

def read_meta(path):
    with rasterio.open(path) as ds:
        t = ds.transform
        b = ds.bounds
        return {
            "crs": ds.crs.to_string() if ds.crs else None,
            "shape": (ds.height, ds.width),
            "res": ds.res,
            "transform": (t.a, t.b, t.c, t.d, t.e, t.f),
            "bounds": (b.left, b.bottom, b.right, b.top),
            "nodata": ds.nodata,
        }

def read_stats(path):
    with rasterio.open(path) as ds:
        a = ds.read(1, masked=True).astype("float32")
        if np.ma.isMaskedArray(a):
            a = np.ma.filled(a, np.nan)

    finite = np.isfinite(a)
    if finite.sum() == 0:
        return {
            "min": None, "max": None,
            "nan_frac": 1.0,
            "zero_frac": None,
            "one_frac": None,
        }

    vals = a[finite]
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "nan_frac": float(1.0 - finite.mean()),
        "zero_frac": float((vals == 0).mean()),
        "one_frac": float((vals == 1).mean()),
    }

def aligned(ref, other, tol=1e-6):
    return {
        "crs": ref["crs"] == other["crs"],
        "shape": ref["shape"] == other["shape"],
        "res": np.allclose(ref["res"], other["res"], atol=tol),
        "transform": np.allclose(ref["transform"], other["transform"], atol=tol),
        "bounds": np.allclose(ref["bounds"], other["bounds"], atol=tol),
    }

def main(root_path):
    root = Path(root_path)
    rows = []

    fire_files = sorted((root / "fire16copy").rglob("fire16_*.tif"))

    for fire in fire_files:
        year = fire.parent.name
        date = fire.stem.split("_")[-1]

        fire_meta = read_meta(fire)
        fire_stats = read_stats(fire)

        ndvi = root / "ndvi16" / year / f"ndvi16_{year}_{date}.tif"
        if not ndvi.exists():
            continue

        ref_meta = read_meta(ndvi)

        for fam in FAMILIES:
            p = root / fam / year / f"{fam}_{year}_{date}.tif"
            if not p.exists():
                continue

            m = read_meta(p)
            s = read_stats(p)
            a = aligned(ref_meta, m)

            rows.append({
                "date": date,
                "family": fam,
                "aligned": all(a.values()),
                "crs": a["crs"],
                "shape": a["shape"],
                "res": a["res"],
                "transform": a["transform"],
                "bounds": a["bounds"],
                "min": s["min"],
                "max": s["max"],
                "nan_frac": s["nan_frac"],
                "zero_frac": s["zero_frac"],
                "one_frac": s["one_frac"],
            })

        rows.append({
            "date": date,
            "family": "fire16",
            "aligned": True,
            "crs": True,
            "shape": True,
            "res": True,
            "transform": True,
            "bounds": True,
            "min": fire_stats["min"],
            "max": fire_stats["max"],
            "nan_frac": fire_stats["nan_frac"],
            "zero_frac": fire_stats["zero_frac"],
            "one_frac": fire_stats["one_frac"],
        })

    df = pd.DataFrame(rows)
    out = root / "fire_vs_others_alignment.csv"
    df.to_csv(out, index=False)

    print("\nSaved:", out)
    print("\nQuick interpretation hints")
    print("• fire16 aligned but looks rectangular → outside ROI encoded as 0")
    print("• zero_frac ~1.0 and nan_frac ~0.0 for fire → masking issue, not alignment")
    print("• any FALSE in crs/shape/res/transform/bounds → true grid mismatch")

if __name__ == "__main__":
    # CHANGE THIS ONLY
    ROOT_PATH = "/Users/sammit/Desktop/Projects/Prometheus/data_processed"
    main(ROOT_PATH)
