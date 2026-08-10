"""MODIS composites (LST 8-day, NDVI 16-day) interpolated to daily fields."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np

from prometheus.config import load_settings
from prometheus.features.warp import fill_nan_nearest, read_to_grid

NDVI_BANDS = ("NDVI", "EVI")
LST_BANDS = ("LST_Day_C", "LST_Night_C")

NDVI_VALID = (-0.2, 1.0)
LST_VALID = (-60.0, 70.0)

_DATE_RE = re.compile(r"(\d{8})")


def product_dir(name: str) -> Path:
    return load_settings().paths.resolve("gee_raw") / name


def composite_dates(name: str, prefix: str) -> list[tuple[date, Path]]:
    folder = product_dir(name)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing GEE folder: {folder}")
    out: list[tuple[date, Path]] = []
    for path in sorted(folder.glob(f"{prefix}_*.tif")):
        m = _DATE_RE.search(path.stem)
        if not m:
            continue
        s = m.group(1)
        try:
            out.append((date(int(s[:4]), int(s[4:6]), int(s[6:8])), path))
        except ValueError:
            continue
    return sorted(out)


def load_year_stack(
    name: str,
    prefix: str,
    year: int,
    valid_range: tuple[float, float],
) -> tuple[list[date], np.ndarray]:
    """Composites for one season, warped to the canonical grid: (n, bands, h, w)."""
    items = [(d, p) for d, p in composite_dates(name, prefix) if d.year == year]
    if not items:
        raise FileNotFoundError(f"No {name} composites for {year}")
    dates: list[date] = []
    frames: list[np.ndarray] = []
    lo, hi = valid_range
    for d, path in items:
        arr, _ = read_to_grid(path, resampling="bilinear")
        arr = np.where((arr >= lo) & (arr <= hi), arr, np.nan)
        dates.append(d)
        frames.append(arr)
    return dates, np.stack(frames, axis=0)


def _ffill_bfill(stack: np.ndarray) -> np.ndarray:
    """Fill NaNs along axis 0 (time), forward then backward, vectorised."""
    n = stack.shape[0]
    steps = np.arange(n).reshape((n,) + (1,) * (stack.ndim - 1))
    valid = np.isfinite(stack)
    out = stack.copy()

    idx = np.where(valid, steps, -1)
    fwd = np.maximum.accumulate(idx, axis=0)
    take = np.where(fwd >= 0, fwd, 0)
    filled = np.take_along_axis(out, take, axis=0)
    out = np.where(fwd >= 0, filled, np.nan)

    valid = np.isfinite(out)
    ridx = np.where(valid, steps, n)
    bwd = np.minimum.accumulate(ridx[::-1], axis=0)[::-1]
    take = np.where(bwd < n, bwd, n - 1)
    filled = np.take_along_axis(out, take, axis=0)
    return np.where(bwd < n, filled, np.nan)


def interpolate_to_daily(
    comp_dates: list[date],
    stack: np.ndarray,
    target_dates: list[date],
) -> np.ndarray:
    """
    Linear-in-time interpolation of composites onto daily dates.

    Cloud gaps are closed along the time axis first, so the interpolation only
    ever runs between observed values. Dates outside the composite span hold the
    nearest composite instead of extrapolating.
    """
    stack = _ffill_bfill(stack)

    comp_ord = np.array([d.toordinal() for d in comp_dates], dtype=np.float64)
    tgt_ord = np.array([d.toordinal() for d in target_dates], dtype=np.float64)

    right = np.searchsorted(comp_ord, tgt_ord, side="right")
    hi = np.clip(right, 1, len(comp_ord) - 1)
    lo = hi - 1
    span = np.maximum(comp_ord[hi] - comp_ord[lo], 1.0)
    w = np.clip((tgt_ord - comp_ord[lo]) / span, 0.0, 1.0).astype(np.float32)

    a = stack[lo]
    b = stack[hi]
    shape = (len(target_dates),) + (1,) * (stack.ndim - 1)
    wt = w.reshape(shape)
    return (a * (1.0 - wt) + b * wt).astype(np.float32)


def daily_year(
    name: str,
    prefix: str,
    year: int,
    target_dates: list[date],
    valid_range: tuple[float, float],
    *,
    climatology: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Daily (n_days, bands, h, w) fields for one season, gaps closed."""
    comp_dates, stack = load_year_stack(name, prefix, year, valid_range)
    daily = interpolate_to_daily(comp_dates, stack, target_dates)

    # Pixels with no valid composite all season: fall back to the multi-year
    # mean for that pixel, then to the nearest neighbour in space.
    bad = ~np.isfinite(daily)
    if bad.any():
        if climatology is not None:
            daily = np.where(bad, climatology[None, ...], daily)
        bad = ~np.isfinite(daily)
        if bad.any():
            for b in range(daily.shape[1]):
                ref = fill_nan_nearest(np.nanmean(daily[:, b], axis=0))
                sl = daily[:, b]
                daily[:, b] = np.where(np.isfinite(sl), sl, ref[None, ...])
    return daily, list(NDVI_BANDS if name == "ndvi" else LST_BANDS)


def pixel_climatology(
    name: str,
    prefix: str,
    years: list[int],
    valid_range: tuple[float, float],
) -> np.ndarray:
    """Per-pixel mean over all composites in all years: (bands, h, w)."""
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for year in years:
        try:
            _, stack = load_year_stack(name, prefix, year, valid_range)
        except FileNotFoundError:
            continue
        finite = np.isfinite(stack)
        vals = np.where(finite, stack, 0.0).sum(axis=0)
        cnt = finite.sum(axis=0).astype(np.float32)
        total = vals if total is None else total + vals
        count = cnt if count is None else count + cnt
    if total is None or count is None:
        raise FileNotFoundError(f"No composites at all for {name}")
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return np.stack([fill_nan_nearest(mean[b]) for b in range(mean.shape[0])], axis=0)


__all__ = [
    "LST_BANDS",
    "LST_VALID",
    "NDVI_BANDS",
    "NDVI_VALID",
    "composite_dates",
    "daily_year",
    "interpolate_to_daily",
    "load_year_stack",
    "pixel_climatology",
    "product_dir",
]
