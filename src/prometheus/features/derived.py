"""Derived predictors: dryness counters, rolling windows, anomalies, fire history."""

from __future__ import annotations

from collections import deque
from datetime import date

import numpy as np

from prometheus import grid
from prometheus.config import load_settings

# A day counts as dry below this much rainfall. 1 mm is the conventional
# threshold: less than that evaporates before it wets the fuel bed.
DRY_DAY_MM = 1.0
# Any measurable rain at all. Kept separate from DRY_DAY_MM so that the two
# counters below describe different things: one tracks fuel drying, the other
# tracks how long since the surface was wetted even slightly.
TRACE_RAIN_MM = 0.1
# Cap for "nothing has burned here yet", in days.
NO_FIRE_SENTINEL = 9999.0

ROLLING_WINDOWS = {
    "precip_7d": ("precip", 7, "sum"),
    "precip_30d": ("precip", 30, "sum"),
    "t2m_max_7d": ("t2m_max", 7, "max"),
    "rh_min_7d": ("rh", 7, "min"),
    "wind_max_7d": ("wind_speed", 7, "max"),
}

DERIVED_WEATHER = (
    "consecutive_dry_days",
    "days_since_rain",
    *ROLLING_WINDOWS.keys(),
)
DERIVED_HISTORY = ("fire_clim", "days_since_fire", "fires_1yr", "fires_3yr", "fires_5yr")
DERIVED_TEMPORAL = ("doy_sin", "doy_cos")
DERIVED_VEGETATION = ("ndvi_anomaly",)


def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum over `window` days ending at each day (inclusive)."""
    c = np.cumsum(arr, axis=0, dtype=np.float32)
    out = c.copy()
    out[window:] = c[window:] - c[:-window]
    return out


def rolling_reduce(arr: np.ndarray, window: int, how: str) -> np.ndarray:
    """Trailing max/min over `window` days ending at each day (inclusive)."""
    if how == "sum":
        return rolling_sum(arr, window)
    fn = np.maximum if how == "max" else np.minimum
    out = arr.astype(np.float32, copy=True)
    for lag in range(1, window):
        out[lag:] = fn(out[lag:], arr[:-lag])
    return out


def dry_spell(precip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Consecutive dry days and days since measurable rain, both ending at day t.

    Both counters restart at the start of each season because the download only
    covers January-May; the first days of January are therefore truncated.
    """
    dry = precip < DRY_DAY_MM
    no_rain = precip < TRACE_RAIN_MM
    t_len = precip.shape[0]
    cdd = np.zeros_like(precip, dtype=np.float32)
    dsr = np.zeros_like(precip, dtype=np.float32)
    run = np.zeros(precip.shape[1:], dtype=np.float32)
    since = np.zeros(precip.shape[1:], dtype=np.float32)
    for t in range(t_len):
        run = np.where(dry[t], run + 1.0, 0.0)
        since = np.where(no_rain[t], since + 1.0, 0.0)
        cdd[t] = run
        dsr[t] = since
    return cdd, dsr


def day_of_year_encoding(dates: list[date]) -> tuple[np.ndarray, np.ndarray]:
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=np.float32)
    angle = 2.0 * np.pi * doy / 365.25
    return np.sin(angle), np.cos(angle)


def season_slot(d: date) -> int:
    """Index of a date within a leap-year Jan 1 - May 31 calendar (0-151).

    Anchoring on a leap year keeps 29 February in its own slot instead of
    shifting every March-May date by one between leap and common years.
    """
    return (d.replace(year=2000) - date(2000, 1, 1)).days


class SeasonAnomaly:
    """
    Per-pixel, per-calendar-day mean of a variable, computed leave-one-year-out.

    Including the target year in its own climatology would leak that year's
    conditions into its anomaly, so each year is scored against the mean of the
    other years only. Accumulating totals once makes that exact and cheap.
    """

    def __init__(self, n_slots: int = 152):
        self.n_slots = n_slots
        self.total: np.ndarray | None = None
        self.count: np.ndarray | None = None
        self.per_year: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def add_year(self, year: int, dates: list[date], values: np.ndarray) -> None:
        slots = np.array([season_slot(d) for d in dates])
        h, w = values.shape[1:]
        acc = np.zeros((self.n_slots, h, w), dtype=np.float32)
        cnt = np.zeros((self.n_slots, 1, 1), dtype=np.float32)
        for i, slot in enumerate(slots):
            acc[slot] += np.nan_to_num(values[i], nan=0.0)
            cnt[slot] += 1.0
        self.per_year[year] = (acc, cnt)
        self.total = acc.copy() if self.total is None else self.total + acc
        self.count = cnt.copy() if self.count is None else self.count + cnt

    def anomaly(self, year: int, dates: list[date], values: np.ndarray) -> np.ndarray:
        if self.total is None or self.count is None:
            raise RuntimeError("SeasonAnomaly has no years accumulated")
        acc, cnt = self.per_year[year]
        others = self.total - acc
        n_others = np.maximum(self.count - cnt, 1.0)
        climatology = others / n_others
        slots = np.array([season_slot(d) for d in dates])
        return (values - climatology[slots]).astype(np.float32)


class FireHistory:
    """
    Streaming fire-history features over the season-only timeline.

    Seasons are contiguous in the data but eight months apart on the calendar,
    so `days_since_fire` is measured in real days rather than cube indices.
    Counts are expressed in whole past seasons, which is the natural unit here
    and avoids windows that dangle into the undownloaded monsoon.
    """

    def __init__(self, shape: tuple[int, int]):
        self.last_fire_ord = np.full(shape, -10**7, dtype=np.int64)
        self.season_totals: deque[np.ndarray] = deque(maxlen=5)

    def advance(self, dates: list[date], fire: np.ndarray) -> None:
        """Update state through a season without materialising its features."""
        for t in range(fire.shape[0]):
            self.last_fire_ord = np.where(
                fire[t] > 0, dates[t].toordinal(), self.last_fire_ord
            )
        self.season_totals.append((fire > 0).sum(axis=0).astype(np.float32))

    def process_year(
        self, dates: list[date], fire: np.ndarray
    ) -> dict[str, np.ndarray]:
        t_len = fire.shape[0]
        shape = fire.shape[1:]

        prev = list(self.season_totals)
        prior = {
            "fires_1yr": _stack_sum(prev[-1:], shape),
            "fires_3yr": _stack_sum(prev[-3:], shape),
            "fires_5yr": _stack_sum(prev[-5:], shape),
        }

        out = {name: np.zeros((t_len, *shape), dtype=np.float32) for name in DERIVED_HISTORY[1:]}
        running = np.zeros(shape, dtype=np.float32)

        for t in range(t_len):
            burning = fire[t] > 0
            ordinal = dates[t].toordinal()
            # Today's detections are available when forecasting tomorrow, so
            # they are folded in before the feature is recorded.
            self.last_fire_ord = np.where(burning, ordinal, self.last_fire_ord)
            running = running + burning

            gap = (ordinal - self.last_fire_ord).astype(np.float32)
            out["days_since_fire"][t] = np.minimum(gap, NO_FIRE_SENTINEL)
            for name, base in prior.items():
                out[name][t] = base + running

        self.season_totals.append((fire > 0).sum(axis=0).astype(np.float32))
        return out


def _stack_sum(items: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not items:
        return np.zeros(shape, dtype=np.float32)
    return np.sum(items, axis=0, dtype=np.float32)


def fire_climatology_slice(dates: list[date]) -> np.ndarray:
    """Out-of-sample day-of-year fire rate (MODIS 2003-2015) for each date."""
    from prometheus.eval.baselines import load_or_build_climatology

    rates = load_or_build_climatology()
    doy = np.array([min(d.timetuple().tm_yday, rates.shape[0] - 1) for d in dates])
    return rates[doy].astype(np.float32)


def rolling_weather(year_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every rolling-window predictor for one season."""
    out: dict[str, np.ndarray] = {}
    for name, (source, window, how) in ROLLING_WINDOWS.items():
        if source not in year_data:
            continue
        out[name] = rolling_reduce(year_data[source], window, how)
    return out


def horizon_labels(fire: np.ndarray, horizons: list[int]) -> dict[str, np.ndarray]:
    """
    Forward-looking labels: did this cell burn within the next H days?

    The last H days of each season have no complete lookahead and are marked
    invalid rather than silently labelled zero.
    """
    t_len = fire.shape[0]
    burning = fire > 0
    out: dict[str, np.ndarray] = {}
    for h in horizons:
        label = np.zeros_like(burning)
        valid = np.zeros(t_len, dtype=bool)
        for t in range(t_len - 1):
            end = min(t + h, t_len - 1)
            label[t] = burning[t + 1 : end + 1].any(axis=0)
            valid[t] = (end - t) == h
        out[f"label_h{h}"] = label
        out[f"valid_h{h}"] = valid
    return out


def default_shape() -> tuple[int, int]:
    return grid.shape()


def horizons() -> list[int]:
    return list(load_settings().modeling.horizons)


__all__ = [
    "DERIVED_HISTORY",
    "DERIVED_TEMPORAL",
    "DERIVED_VEGETATION",
    "DERIVED_WEATHER",
    "DRY_DAY_MM",
    "FireHistory",
    "NO_FIRE_SENTINEL",
    "ROLLING_WINDOWS",
    "SeasonAnomaly",
    "day_of_year_encoding",
    "default_shape",
    "dry_spell",
    "fire_climatology_slice",
    "horizon_labels",
    "horizons",
    "rolling_reduce",
    "rolling_sum",
    "rolling_weather",
    "season_slot",
]
