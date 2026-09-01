"""FIRMS download, clean, rasterize — Day 2 fire labels."""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from scipy import ndimage
from tqdm import tqdm

from prometheus.config import load_settings, project_root
from prometheus import grid

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{day_range}/{date}"
# Official docs (firms.modaps.eosdis.nasa.gov/api/area/): DAY_RANGE is 1..5 only.
# Using 6–10 returns HTTP 400 Bad Request.
MAX_DAY_RANGE = 5


# ---------------------------------------------------------------------------
# Paths / API key
# ---------------------------------------------------------------------------

def firms_dir() -> Path:
    p = load_settings().paths.resolve("firms_raw")
    p.mkdir(parents=True, exist_ok=True)
    return p


def chunks_dir() -> Path:
    p = firms_dir() / "chunks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_map_key(explicit: str | None = None) -> str:
    """Resolve MAP_KEY from arg, env, or gitignored local file."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("MAP_KEY")
    if env:
        return env.strip()
    for candidate in (
        firms_dir() / ".map_key",
        project_root() / "firms_api_key.txt",
        project_root() / ".env",
    ):
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8").strip()
        if candidate.name == ".env":
            for line in text.splitlines():
                if line.startswith("FIRMS_MAP_KEY=") or line.startswith("MAP_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        elif text:
            # first non-empty / non-comment line
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    raise RuntimeError(
        "No FIRMS MAP_KEY found. Set FIRMS_MAP_KEY env var or write it to "
        f"{firms_dir() / '.map_key'} (gitignored under data/)."
    )


def save_map_key(key: str) -> Path:
    path = firms_dir() / ".map_key"
    path.write_text(key.strip() + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceWindow:
    source: str
    start: date
    end: date  # inclusive


def default_source_windows(*, season_only: bool = True) -> list[SourceWindow]:
    """
    Download windows from the plan.

    season_only=True (default): only Jan–May each year — enough for labels
    and pre-monsoon climatology, far fewer API transactions.
    season_only=False: full calendar years (heavier; use only if needed).
    """
    s = load_settings().season
    windows: list[SourceWindow] = []

    def add(source: str, y0: int, y1: int) -> None:
        for y in range(y0, y1 + 1):
            if season_only:
                windows.append(
                    SourceWindow(
                        source,
                        date(y, s.start_month, s.start_day),
                        date(y, s.end_month, s.end_day),
                    )
                )
            else:
                windows.append(SourceWindow(source, date(y, 1, 1), date(y, 12, 31)))

    years = load_settings().years
    y0, y1 = years.train_start, years.train_end
    add("VIIRS_SNPP_SP", y0, y1)
    # NOAA-20 VIIRS SP starts ~2018
    add("VIIRS_NOAA20_SP", max(y0, 2018), y1)
    # MODIS includes climatology years 2003–(y0-1) plus label years through train_end
    clim0 = min(years.climatology) if years.climatology else 2003
    add("MODIS_SP", clim0, y1)
    return windows


def _date_chunks(start: date, end: date, span: int = MAX_DAY_RANGE) -> list[tuple[date, int]]:
    """
    Yield (window_start, day_range) covering [start, end] inclusive.

    FIRMS Area API with DATE returns data for:
        DATE .. DATE + (DAY_RANGE - 1)
    so DATE is the *start* of the window (see /api/area docs).
    """
    if span < 1 or span > MAX_DAY_RANGE:
        raise ValueError(f"day_range must be 1..{MAX_DAY_RANGE}, got {span}")
    out: list[tuple[date, int]] = []
    cur = start
    while cur <= end:
        day_range = min(span, (end - cur).days + 1)
        out.append((cur, day_range))
        cur = cur + timedelta(days=day_range)
    return out


def _bbox_str() -> str:
    """west,south,east,north — keep modest precision (API path-friendly)."""
    s = load_settings()
    w, s_lat, e, n = s.labels.bbox
    return f"{float(w):.4f},{float(s_lat):.4f},{float(e):.4f},{float(n):.4f}"


def _chunk_path(source: str, start_date: date, day_range: int) -> Path:
    return chunks_dir() / f"{source}_{start_date.isoformat()}_d{day_range}.csv"


def download_chunk(
    map_key: str,
    source: str,
    start_date: date,
    day_range: int,
    *,
    sleep_s: float = 0.25,
    timeout: int = 120,
    force: bool = False,
) -> Path:
    """Download one FIRMS area CSV (cached). Returns path to local CSV."""
    if day_range < 1 or day_range > MAX_DAY_RANGE:
        raise ValueError(f"day_range must be 1..{MAX_DAY_RANGE}, got {day_range}")

    path = _chunk_path(source, start_date, day_range)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path

    url = FIRMS_AREA_URL.format(
        key=map_key,
        source=source,
        bbox=_bbox_str(),
        day_range=day_range,
        date=start_date.isoformat(),
    )
    resp = requests.get(url, timeout=timeout)
    if resp.status_code >= 400:
        # Surfaced body helps debug MAP_KEY / date / source issues
        raise RuntimeError(
            f"FIRMS HTTP {resp.status_code} for {source} start={start_date} "
            f"day_range={day_range}: {resp.text.strip()[:300]}\nURL: {url}"
        )
    text = resp.text.strip()
    lower = text.lower()
    if lower.startswith("invalid") or "exceeded" in lower or (
        "error" in lower[:120] and "latitude" not in lower[:200]
    ):
        raise RuntimeError(f"FIRMS API error for {source} {start_date}: {text[:300]}")
    if not text or text == "No data" or "no data" in lower[:40]:
        path.write_text("latitude,longitude,acq_date\n", encoding="utf-8")
    else:
        path.write_text(resp.text, encoding="utf-8")
    if sleep_s > 0:
        time.sleep(sleep_s)
    return path


def download_all(
    map_key: str | None = None,
    windows: Iterable[SourceWindow] | None = None,
    *,
    sleep_s: float = 0.25,
    force: bool = False,
) -> list[Path]:
    key = resolve_map_key(map_key)
    windows = list(windows or default_source_windows())
    jobs: list[tuple[str, date, int]] = []
    for w in windows:
        for start_date, day_range in _date_chunks(w.start, w.end):
            jobs.append((w.source, start_date, day_range))

    paths: list[Path] = []
    errors: list[str] = []
    for source, start_date, day_range in tqdm(jobs, desc="FIRMS download"):
        try:
            paths.append(
                download_chunk(
                    key, source, start_date, day_range, sleep_s=sleep_s, force=force
                )
            )
        except Exception as exc:
            errors.append(str(exc))
            # keep going — partial cache is better than aborting at job 1
            tqdm.write(f"[warn] skip {source} {start_date} d{day_range}: {exc}")

    if errors and not paths:
        raise RuntimeError(
            "All FIRMS downloads failed. First error:\n" + errors[0]
        )
    if errors:
        print(f"[warn] {len(errors)} chunk(s) failed; continuing with {len(paths)} ok.")
    return paths


# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------

_REQUIRED_COLS = {"latitude", "longitude", "acq_date"}


def _read_chunk_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if not _REQUIRED_COLS.issubset(set(df.columns)):
        return pd.DataFrame()
    df["source_file"] = path.name
    # instrument / satellite hints from filename
    name = path.name.upper()
    if "VIIRS_NOAA20" in name or "VIIRS_NOAA21" in name:
        df["collection"] = "VIIRS_NOAA20_SP" if "NOAA20" in name else "VIIRS_NOAA21_SP"
    elif "VIIRS_SNPP" in name:
        df["collection"] = "VIIRS_SNPP_SP"
    elif "MODIS" in name:
        df["collection"] = "MODIS_SP"
    else:
        df["collection"] = "UNKNOWN"
    return df


def load_raw_chunks(paths: Iterable[Path] | None = None) -> pd.DataFrame:
    if paths is None:
        paths = sorted(chunks_dir().glob("*.csv"))
    frames = []
    for p in paths:
        df = _read_chunk_csv(p)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean_firms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Day-2 cleaning rules:
    - MODIS: confidence >= 50
    - VIIRS: confidence in {nominal, high} (or high numeric if present)
    - drop type != 0 when column exists
    - dedupe on (round(lat,4), round(lon,4), acq_date, satellite)
    """
    settings = load_settings()
    if df.empty:
        return df.copy()

    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["acq_date"] = pd.to_datetime(out["acq_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["latitude", "longitude", "acq_date"])

    # BBox clip (defensive)
    w, s, e, n = settings.labels.bbox
    out = out[
        (out["longitude"] >= w)
        & (out["longitude"] <= e)
        & (out["latitude"] >= s)
        & (out["latitude"] <= n)
    ]

    # type filter (0 = presumed vegetation fire)
    if settings.labels.drop_type_nonzero and "type" in out.columns:
        t = pd.to_numeric(out["type"], errors="coerce")
        # keep missing type (treat as 0) or type == 0
        out = out[t.isna() | (t == 0)]

    # confidence by sensor family
    if "confidence" in out.columns:
        conf = out["confidence"]
        conf_str = conf.astype(str).str.strip().str.lower()
        conf_num = pd.to_numeric(conf, errors="coerce")

        is_viirs = out["collection"].astype(str).str.contains("VIIRS", case=False, na=False)
        is_modis = out["collection"].astype(str).str.contains("MODIS", case=False, na=False)

        allowed_viirs = {c.lower() for c in settings.labels.viirs_confidence}
        # also accept common single-letter codes some archives use: n/h
        allowed_viirs |= {"n", "h", "nominal", "high"}

        keep_viirs = is_viirs & (
            conf_str.isin(allowed_viirs)
            | conf_str.isin({"nominal", "high", "n", "h"})
            # numeric fallback sometimes used
            | (conf_num >= 50)
        )
        keep_modis = is_modis & (conf_num >= float(settings.labels.modis_confidence_min))
        keep_other = ~(is_viirs | is_modis)
        out = out[keep_viirs | keep_modis | keep_other]

    # satellite id for dedupe
    if "satellite" not in out.columns:
        out["satellite"] = out.get("collection", "UNK")
    out["satellite"] = out["satellite"].astype(str).fillna("UNK")

    out["lat_r"] = out["latitude"].round(4)
    out["lon_r"] = out["longitude"].round(4)
    out = out.drop_duplicates(subset=["lat_r", "lon_r", "acq_date", "satellite"])
    out = out.drop(columns=["lat_r", "lon_r"])

    out = out.sort_values(["acq_date", "latitude", "longitude"]).reset_index(drop=True)
    return out


def season_filter(df: pd.DataFrame) -> pd.DataFrame:
    months = set(load_settings().season.months)
    if df.empty:
        return df
    m = df["acq_date"].dt.month
    return df[m.isin(months)].reset_index(drop=True)


def detection_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["year"] = tmp["acq_date"].dt.year
    tmp["month"] = tmp["acq_date"].dt.month
    return pd.crosstab(tmp["year"], tmp["month"]).sort_index()


# ---------------------------------------------------------------------------
# Rasterize daily cube
# ---------------------------------------------------------------------------

def _season_dates(year_start: int, year_end: int) -> list[np.datetime64]:
    s = load_settings().season
    dates: list[np.datetime64] = []
    for y in range(year_start, year_end + 1):
        d0 = date(y, s.start_month, s.start_day)
        d1 = date(y, s.end_month, s.end_day)
        cur = d0
        while cur <= d1:
            dates.append(np.datetime64(cur.isoformat()))
            cur += timedelta(days=1)
    return dates


def load_archive_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    """Load full FIRMS-style archives (not API chunk cache format)."""
    frames = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if not _REQUIRED_COLS.issubset(set(df.columns)):
            continue
        # Infer collection from instrument/filename
        name = path.name.upper()
        if "instrument" in df.columns:
            inst = df["instrument"].astype(str).str.upper()
            df["collection"] = np.where(
                inst.str.contains("VIIRS"),
                "VIIRS_SNPP_SP",
                "MODIS_SP",
            )
            # refine NOAA-20 if satellite column says N20 / N / 1
            if "satellite" in df.columns:
                sat = df["satellite"].astype(str).str.upper()
                is_n20 = sat.str.contains("N20") | sat.isin(["1", "N20", "J1"])
                df.loc[is_n20 & inst.str.contains("VIIRS"), "collection"] = "VIIRS_NOAA20_SP"
        elif "VIIRS" in name:
            df["collection"] = "VIIRS_SNPP_SP"
        else:
            df["collection"] = "MODIS_SP"
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def rasterize_daily(
    df: pd.DataFrame,
    *,
    year_start: int | None = None,
    year_end: int | None = None,
    dilate_pixels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (T, H, W) uint8 fire cube for Jan–May years.

    Points outside Nepal mask are ignored. Outside-mask cells stay 0.
    Vectorized snap + optional morphological dilate.
    """
    settings = load_settings()
    year_start = year_start if year_start is not None else settings.years.train_start
    year_end = year_end if year_end is not None else settings.years.train_end
    dilate = settings.labels.dilate_pixels if dilate_pixels is None else dilate_pixels

    times = np.array(_season_dates(year_start, year_end))
    h, w = grid.shape()
    cube = np.zeros((len(times), h, w), dtype=np.uint8)
    mask = grid.nepal_mask()
    t = grid.transform()

    if df.empty:
        return cube, times

    work = df.copy()
    work["acq_date"] = pd.to_datetime(work["acq_date"]).dt.normalize()
    work = season_filter(work)
    work = work[
        (work["acq_date"].dt.year >= year_start) & (work["acq_date"].dt.year <= year_end)
    ]
    if work.empty:
        return cube, times

    # map each date → time index
    time_index = {pd.Timestamp(tt).to_datetime64(): i for i, tt in enumerate(times)}
    # also Timestamp keys
    time_index.update({pd.Timestamp(tt).normalize(): i for i, tt in enumerate(times)})

    xs = work["longitude"].to_numpy(dtype=np.float64)
    ys = work["latitude"].to_numpy(dtype=np.float64)
    # affine with b=d=0 (north-up): col=(x-c)/a , row=(y-f)/e
    cols = np.floor((xs - t.c) / t.a).astype(np.int32)
    rows = np.floor((ys - t.f) / t.e).astype(np.int32)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    # inside Nepal
    valid[valid] &= mask[rows[valid], cols[valid]]

    dates = work["acq_date"].to_numpy(dtype="datetime64[ns]")
    for i in np.where(valid)[0]:
        ti = time_index.get(pd.Timestamp(dates[i]).normalize())
        if ti is None:
            # try raw numpy datetime64
            ti = time_index.get(dates[i])
        if ti is None:
            continue
        cube[ti, rows[i], cols[i]] = 1

    if dilate and dilate > 0:
        structure = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=bool)
        for ti in range(cube.shape[0]):
            if cube[ti].any():
                dilated = ndimage.binary_dilation(cube[ti].astype(bool), structure=structure)
                dilated &= mask
                cube[ti] = dilated.astype(np.uint8)

    cube[:, ~mask] = 0
    return cube, times


def save_fire_zarr(
    cube: np.ndarray,
    times: np.ndarray,
    path: Path | None = None,
) -> Path:
    """Write fire_daily.zarr with attrs for alignment audit."""
    import shutil

    import xarray as xr

    settings = load_settings()
    path = path or (settings.paths.resolve("cube") / "fire_daily.zarr")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)

    h, w = grid.shape()
    t = grid.transform()
    cols = np.arange(w)
    rows = np.arange(h)
    xs = t.c + (cols + 0.5) * t.a + (0 + 0.5) * t.b
    ys = t.f + (0 + 0.5) * t.d + (rows + 0.5) * t.e

    da = xr.DataArray(
        cube,
        dims=("time", "y", "x"),
        coords={
            "time": times.astype("datetime64[ns]"),
            "y": ys,
            "x": xs,
        },
        name="fire",
        attrs={
            "long_name": "daily active-fire label",
            "description": "1 = fire detection (dilated), 0 = no fire; 0 outside Nepal",
            "crs": settings.grid.crs,
            "transform": [float(x) for x in t[:6]],
            "height": h,
            "width": w,
            "dilate_pixels": settings.labels.dilate_pixels,
            "season_months": settings.season.months,
            "grid_aligned_to": "nepal_mask_1km_roiAligned.tif",
        },
    )
    ds = da.to_dataset()
    ds.attrs["created"] = datetime.utcnow().isoformat() + "Z"
    ds.attrs["prometheus_version"] = settings.project.version
    ds.to_zarr(path, mode="w")
    return path


def assert_cube_alignment(cube: np.ndarray) -> dict:
    """
    Alignment / integrity checks for the fire cube.
    Raises AssertionError on failure; returns summary stats.
    """
    settings = load_settings()
    h, w = settings.grid.shape
    assert cube.ndim == 3, f"expected 3D cube, got {cube.ndim}D"
    assert cube.shape[1] == h and cube.shape[2] == w, (
        f"spatial shape {cube.shape[1:]} != grid {(h, w)}"
    )
    assert cube.dtype == np.uint8, f"dtype {cube.dtype} != uint8"
    assert set(np.unique(cube)).issubset({0, 1}), "cube must be binary 0/1"

    mask = grid.nepal_mask()
    outside = cube[:, ~mask]
    n_outside = int(outside.sum())
    assert n_outside == 0, f"found {n_outside} fire pixels outside Nepal mask"

    return {
        "shape": list(cube.shape),
        "total_fire_pixels": int(cube.sum()),
        "days_with_fire": int((cube.sum(axis=(1, 2)) > 0).sum()),
        "outside_mask_fire_pixels": n_outside,
        "mask_valid_pixels": int(mask.sum()),
    }


def build_pipeline(
    map_key: str | None = None,
    *,
    skip_download: bool = False,
    sleep_s: float = 0.25,
    archive_paths: Iterable[Path] | None = None,
    season_only_download: bool = True,
) -> dict:
    """
    End-to-end Day-2: download → clean → rasterize → zarr → checks.
    Optionally merge local archive CSVs (legacy / manual downloads).
    """
    settings = load_settings()
    if map_key:
        save_map_key(map_key)

    download_error = None
    if not skip_download:
        try:
            download_all(
                map_key=map_key,
                windows=default_source_windows(season_only=season_only_download),
                sleep_s=sleep_s,
            )
        except Exception as exc:  # network / quota
            download_error = str(exc)
            print(f"[warn] FIRMS download failed: {exc}")

    frames = []
    raw_chunks = load_raw_chunks()
    if not raw_chunks.empty:
        frames.append(raw_chunks)

    # default archives under data/raw/firms/archives/ (seed MODIS export, manual drops)
    archives_dir = firms_dir() / "archives"
    default_archives: list[Path] = []
    if archives_dir.is_dir():
        default_archives.extend(sorted(archives_dir.glob("*.csv")))
    if archive_paths is not None:
        default_archives.extend(list(archive_paths))

    if default_archives:
        arch = load_archive_csvs(default_archives)
        if not arch.empty:
            frames.append(arch)

    if not frames:
        raise RuntimeError(
            "No FIRMS data available. Run with network download or place CSVs in "
            f"{firms_dir() / 'archives'}"
        )

    raw = pd.concat(frames, ignore_index=True)
    cleaned = clean_firms(raw)
    cleaned_path = firms_dir() / "firms_clean_points.parquet"
    try:
        cleaned.to_parquet(cleaned_path, index=False)
    except Exception:
        cleaned_path = firms_dir() / "firms_clean_points.csv"
        cleaned.to_csv(cleaned_path, index=False)
    cleaned.to_csv(firms_dir() / "firms_clean_points.csv", index=False)

    season = season_filter(cleaned)
    y0, y1 = settings.years.train_start, settings.years.train_end
    season_train = season[
        (season["acq_date"].dt.year >= y0) & (season["acq_date"].dt.year <= y1)
    ]

    table = detection_table(season_train)
    table_path = firms_dir() / "detection_table_year_month.csv"
    table.to_csv(table_path)

    cube, times = rasterize_daily(cleaned)
    align = assert_cube_alignment(cube)
    zarr_path = save_fire_zarr(cube, times)
    import xarray as xr

    ds = xr.open_zarr(zarr_path)
    fire = ds["fire"].values
    align2 = assert_cube_alignment(fire.astype(np.uint8))

    # Collection breakdown
    by_coll = (
        season_train.groupby("collection").size().to_dict()
        if "collection" in season_train.columns and not season_train.empty
        else {}
    )

    n_season = int(len(season_train))
    report = {
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "season_train_rows": n_season,
        # legacy key kept for any notebooks that still read it
        "season_2016_2025_rows": n_season,
        "train_years": f"{y0}-{y1}",
        "by_collection_season": by_coll,
        "detection_table_path": str(table_path),
        "cleaned_path": str(cleaned_path),
        "zarr_path": str(zarr_path),
        "alignment": align,
        "alignment_reloaded": align2,
        "times": int(len(times)),
        "passes_120k": n_season > 120_000,
        "download_error": download_error,
        "archives_used": [str(p) for p in default_archives],
    }
    report_path = firms_dir() / "day2_build_report.json"
    import json

    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
