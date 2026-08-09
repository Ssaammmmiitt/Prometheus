"""Baseline risk models: climatology and persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from prometheus import grid
from prometheus.config import load_settings, project_root


def fire_cube_path() -> Path:
    return load_settings().paths.resolve("cube") / "fire_daily.zarr"


def load_fire_cube() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load uint8 fire cube (T,H,W) and times."""
    path = fire_cube_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing fire cube at {path}. Run Day-2 first.")
    ds = xr.open_zarr(path)
    fire = np.asarray(ds["fire"].values, dtype=np.uint8)
    times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).normalize()
    return fire, times


def clean_points_path() -> Path:
    firms = load_settings().paths.resolve("firms_raw")
    for name in ("firms_clean_points.parquet", "firms_clean_points.csv"):
        p = firms / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No clean points under {firms}")


def build_modis_climatology(
    *,
    years: list[int] | None = None,
    temporal_half_window: int = 7,
    spatial_sigma: float = 1.0,
    save: bool = True,
) -> np.ndarray:
    """
    Day-of-year fire probability map from MODIS 2003–2015 (config climatology years).

    Returns array shape (366, H, W) float32 in [0, 1], zero outside Nepal mask.
    Index 0 unused; doy 1..366 used (numpy dayofyear).
    """
    settings = load_settings()
    years = years if years is not None else list(settings.years.climatology)
    h, w = grid.shape()
    mask = grid.nepal_mask()
    t = grid.transform()

    path = clean_points_path()
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.normalize()
    df = df[df["acq_date"].dt.year.isin(years)]
    if "collection" in df.columns:
        df = df[df["collection"].astype(str).str.contains("MODIS", case=False, na=False)]
    # season months only (matches prediction season)
    df = df[df["acq_date"].dt.month.isin(settings.season.months)]

    counts = np.zeros((367, h, w), dtype=np.float64)  # doy 1..366
    if not df.empty:
        xs = df["longitude"].to_numpy(dtype=np.float64)
        ys = df["latitude"].to_numpy(dtype=np.float64)
        cols = np.floor((xs - t.c) / t.a).astype(np.int32)
        rows = np.floor((ys - t.f) / t.e).astype(np.int32)
        doys = df["acq_date"].dt.dayofyear.to_numpy()
        years_arr = df["acq_date"].dt.year.to_numpy()
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        valid[valid] &= mask[rows[valid], cols[valid]]
        tmp = pd.DataFrame(
            {
                "year": years_arr[valid],
                "doy": doys[valid],
                "r": rows[valid],
                "c": cols[valid],
            }
        ).drop_duplicates()
        np.add.at(
            counts,
            (tmp["doy"].to_numpy(), tmp["r"].to_numpy(), tmp["c"].to_numpy()),
            1.0,
        )

    n_years = float(max(len(years), 1))
    rates = (counts / n_years).astype(np.float32)

    # Temporal smooth ± half window along day-of-year (vectorized)
    if temporal_half_window and temporal_half_window > 0:
        kernel = 2 * temporal_half_window + 1
        body = rates[1:367]  # (366, H, W)
        sm = ndimage.uniform_filter1d(
            body.astype(np.float64), size=kernel, axis=0, mode="nearest"
        )
        rates[1:367] = sm.astype(np.float32)

    # Spatial Gaussian per doy (only on valid mask)
    if spatial_sigma and spatial_sigma > 0:
        for doy in range(1, 367):
            layer = rates[doy]
            if not layer.any():
                continue
            sm = ndimage.gaussian_filter(layer.astype(np.float64), sigma=spatial_sigma)
            sm = np.clip(sm, 0.0, 1.0)
            sm[~mask] = 0.0
            rates[doy] = sm.astype(np.float32)

    rates[:, ~mask] = 0.0
    rates = np.clip(rates, 0.0, 1.0)

    if save:
        out = load_settings().paths.resolve("cube") / "climatology_doy.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            rates=rates,
            years=np.asarray(years, dtype=np.int32),
            temporal_half_window=temporal_half_window,
            spatial_sigma=spatial_sigma,
        )
    return rates


def load_or_build_climatology(force: bool = False) -> np.ndarray:
    path = load_settings().paths.resolve("cube") / "climatology_doy.npz"
    if path.is_file() and not force:
        data = np.load(path)
        return data["rates"]
    return build_modis_climatology(save=True)


def climatology_for_times(rates: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    """(T, H, W) predictions by day-of-year lookup."""
    doys = times.dayofyear.to_numpy()
    return rates[doys].astype(np.float32)


def persistence_scores(
    fire: np.ndarray,
    times: pd.DatetimeIndex,
    *,
    lookback_days: int = 7,
) -> np.ndarray:
    """
    For each day t: max fire in this cell or its 8 neighbours over the past
    1..lookback_days, restricted to the same calendar year (no May→Jan bleed).

    Output float32 (T,H,W) with 0/1 values.
    """
    t_len, h, w = fire.shape
    out = np.zeros((t_len, h, w), dtype=np.float32)
    fire_f = fire.astype(bool)
    years = times.year.to_numpy()
    mask = grid.nepal_mask()

    for year in np.unique(years):
        idx = np.where(years == year)[0]
        if idx.size == 0:
            continue
        block = fire_f[idx]  # (Ty, H, W)
        ty = block.shape[0]
        # causal OR over past lookback frames along axis 0
        past_any = np.zeros((ty, h, w), dtype=bool)
        for i in range(1, ty):
            start = max(0, i - lookback_days)
            # block[start:i] are strictly previous days in this year's seasonal cube
            past_any[i] = block[start:i].any(axis=0)
        # 3×3 spatial max (center + 8 neighbours) — batch one filter per day via loop is OK;
        # vectorize with maximum_filter on reshaped stack
        # Apply spatial max day-wise in one pass using 3D filter only in spatial dims
        for i in range(ty):
            if not past_any[i].any():
                continue
            nb = ndimage.maximum_filter(past_any[i].astype(np.uint8), size=3).astype(bool)
            nb &= mask
            out[idx[i]] = nb.astype(np.float32)
    return out


def year_indices(times: pd.DatetimeIndex, year: int) -> np.ndarray:
    return np.where(times.year == year)[0]
