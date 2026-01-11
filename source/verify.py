"""
Local raster verifier for Prometheus project structure

Uses your local-style paths:
  ROOT = /Users/sammit/Desktop/Projects/Prometheus
  DATA_NORM = ROOT / data_processed_normalized
  DATA_FIRE = ROOT / data_processed / fire16
  REPORT_DIR = ROOT / reports / dataset

Outputs:
  REPORT_DIR / raster_audit.csv
  REPORT_DIR / raster_audit_summary.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm


# =========================
# LOCAL CONFIG (your style)
# =========================
ROOT = Path("/Users/sammit/Desktop/Projects/Prometheus").resolve()

DATA_NORM = ROOT / "data_processed_normalized"
DATA_FIRE = ROOT / "data_processed" / "fire16"

REPORT_DIR = ROOT / "reports" / "dataset"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = REPORT_DIR / "raster_audit.csv"
OUT_JSON = REPORT_DIR / "raster_audit_summary.json"


# =========================
# STRUCT
# =========================
@dataclass
class RasterAuditRow:
    kind: str
    var: str
    year: str
    filename: str
    path: str

    ok: bool
    error: str

    dtype: str
    height: int
    width: int

    crs: str
    nodata: str

    min_val: float
    max_val: float
    nan_count: int
    inf_count: int

    eq_nodata_count: int
    eq_minus9999_count: int

    total_pixels: int
    suspect_pixels: int
    suspect_pct: float


def collect_tifs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.tif") if p.is_file()])


def parse_var_year_from_path(p: Path, data_root: Path, fire_root: Path) -> Tuple[str, str, str]:
    s = str(p)
    if fire_root in p.parents or str(fire_root) in s:
        kind = "fire"
        var = "fire16"
        year = "unknown"
        for part in p.parts[::-1]:
            if part.isdigit() and len(part) == 4:
                year = part
                break
        return kind, var, year

    kind = "data"
    var = "unknown"
    year = "static"
    try:
        rel = p.relative_to(data_root)
        if len(rel.parts) >= 1:
            var = rel.parts[0]
        if len(rel.parts) >= 2:
            if rel.parts[1].isdigit() and len(rel.parts[1]) == 4:
                year = rel.parts[1]
            elif rel.parts[1] == "static":
                year = "static"
    except Exception:
        pass

    return kind, var, year


def audit_one_tif(p: Path, kind: str, var: str, year: str) -> RasterAuditRow:
    try:
        with rasterio.open(p) as src:
            arr = src.read(1)
            dtype = str(arr.dtype)
            h, w = arr.shape
            total = int(arr.size)

            nodata = src.nodata
            nodata_str = str(nodata) if nodata is not None else ""

            if np.issubdtype(arr.dtype, np.floating):
                nan_count = int(np.isnan(arr).sum())
                inf_count = int(np.isinf(arr).sum())
                finite = np.isfinite(arr)
                if finite.any():
                    min_val = float(arr[finite].min())
                    max_val = float(arr[finite].max())
                else:
                    min_val = float("nan")
                    max_val = float("nan")
            else:
                nan_count = 0
                inf_count = 0
                min_val = float(arr.min())
                max_val = float(arr.max())

            eq_minus9999 = int((arr == -9999).sum())
            if nodata is None:
                eq_nodata = 0
            else:
                eq_nodata = int((arr == nodata).sum())

            suspect = eq_minus9999 + eq_nodata + nan_count + inf_count
            suspect_pct = 100.0 * suspect / max(total, 1)

            crs = str(src.crs) if src.crs is not None else ""

            return RasterAuditRow(
                kind=kind,
                var=var,
                year=year,
                filename=p.name,
                path=str(p),
                ok=True,
                error="",
                dtype=dtype,
                height=int(h),
                width=int(w),
                crs=crs,
                nodata=nodata_str,
                min_val=min_val,
                max_val=max_val,
                nan_count=nan_count,
                inf_count=inf_count,
                eq_nodata_count=eq_nodata,
                eq_minus9999_count=eq_minus9999,
                total_pixels=total,
                suspect_pixels=int(suspect),
                suspect_pct=float(suspect_pct),
            )

    except RasterioIOError as e:
        return RasterAuditRow(
            kind=kind,
            var=var,
            year=year,
            filename=p.name,
            path=str(p),
            ok=False,
            error=f"RasterioIOError: {e}",
            dtype="",
            height=0,
            width=0,
            crs="",
            nodata="",
            min_val=float("nan"),
            max_val=float("nan"),
            nan_count=0,
            inf_count=0,
            eq_nodata_count=0,
            eq_minus9999_count=0,
            total_pixels=0,
            suspect_pixels=0,
            suspect_pct=float("nan"),
        )
    except Exception as e:
        return RasterAuditRow(
            kind=kind,
            var=var,
            year=year,
            filename=p.name,
            path=str(p),
            ok=False,
            error=f"Exception: {e}",
            dtype="",
            height=0,
            width=0,
            crs="",
            nodata="",
            min_val=float("nan"),
            max_val=float("nan"),
            nan_count=0,
            inf_count=0,
            eq_nodata_count=0,
            eq_minus9999_count=0,
            total_pixels=0,
            suspect_pixels=0,
            suspect_pct=float("nan"),
        )


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "files_total": int(len(df)),
        "files_ok": int(df["ok"].sum()) if len(df) else 0,
        "files_failed": int((~df["ok"]).sum()) if len(df) else 0,
    }
    if len(df) == 0:
        return summary

    ok_df = df[df["ok"] == True].copy()

    def agg(block: pd.DataFrame) -> Dict[str, Any]:
        if len(block) == 0:
            return {}
        return {
            "files": int(len(block)),
            "suspect_pixels_total": int(block["suspect_pixels"].sum()),
            "eq_minus9999_total": int(block["eq_minus9999_count"].sum()),
            "eq_nodata_total": int(block["eq_nodata_count"].sum()),
            "nan_total": int(block["nan_count"].sum()),
            "inf_total": int(block["inf_count"].sum()),
            "max_suspect_pct": float(block["suspect_pct"].max()),
            "mean_suspect_pct": float(block["suspect_pct"].mean()),
        }

    summary["data"] = agg(ok_df[ok_df["kind"] == "data"])
    summary["fire"] = agg(ok_df[ok_df["kind"] == "fire"])

    worst = ok_df.sort_values("suspect_pct", ascending=False).head(30)
    summary["worst_30_by_suspect_pct"] = worst[
        ["kind", "var", "year", "filename", "suspect_pct", "eq_minus9999_count", "eq_nodata_count", "nan_count", "inf_count", "path"]
    ].to_dict(orient="records")

    minus = ok_df[ok_df["eq_minus9999_count"] > 0].sort_values("eq_minus9999_count", ascending=False).head(30)
    summary["top_30_by_minus9999_count"] = minus[
        ["kind", "var", "year", "filename", "eq_minus9999_count", "suspect_pct", "path"]
    ].to_dict(orient="records")

    return summary


def main():
    data_files = collect_tifs(DATA_NORM)
    fire_files = collect_tifs(DATA_FIRE)
    all_files = data_files + fire_files

    if len(all_files) == 0:
        raise SystemExit("No .tif files found. Check DATA_NORM and DATA_FIRE paths.")

    rows: List[RasterAuditRow] = []
    for p in tqdm(all_files, desc="Auditing rasters"):
        kind, var, year = parse_var_year_from_path(p, DATA_NORM, DATA_FIRE)
        rows.append(audit_one_tif(p, kind=kind, var=var, year=year))

    df = pd.DataFrame([asdict(r) for r in rows]).sort_values(
        ["kind", "var", "year", "filename"]
    ).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(summarize(df), indent=2))

    print("Done")
    print("DATA_NORM:", DATA_NORM)
    print("DATA_FIRE:", DATA_FIRE)
    print("CSV:", OUT_CSV)
    print("Summary:", OUT_JSON)

    ok_df = df[df["ok"] == True]
    print("Files total:", len(df), "ok:", int(ok_df.shape[0]), "failed:", int((~df["ok"]).sum()))

    minus = ok_df[ok_df["eq_minus9999_count"] > 0]
    print("Files containing exact -9999 pixels:", int(minus.shape[0]))
    if len(minus) > 0:
        top = minus.sort_values("eq_minus9999_count", ascending=False).head(10)
        print("\nTop 10 files by -9999 count:")
        for _, r in top.iterrows():
            print(int(r["eq_minus9999_count"]), "pct", f"{r['suspect_pct']:.3f}", r["path"])


if __name__ == "__main__":
    main()
