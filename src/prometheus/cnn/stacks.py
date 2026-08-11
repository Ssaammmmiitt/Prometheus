"""Dense per-season feature stacks for convolutional models.

The tabular path only ever materialises forest cells, but a CNN needs whole
rasters, so each season is cached once as (T, C, H, W) float16 on disk and read
one day-plane at a time during training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prometheus import grid
from prometheus.config import load_settings
from prometheus.features import derived
from prometheus.features import table as ftable

FLOAT_DTYPE = np.float16


def stacks_dir() -> Path:
    path = load_settings().paths.resolve("cube") / "stacks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_path(year: int) -> Path:
    return stacks_dir() / f"feat_{year}.npy"


def label_path(year: int, horizon: int) -> Path:
    return stacks_dir() / f"label_{year}_h{horizon}.npy"


def meta_path(year: int) -> Path:
    return stacks_dir() / f"meta_{year}.json"


def is_cached(year: int, horizons: list[int]) -> bool:
    return (
        feature_path(year).is_file()
        and meta_path(year).is_file()
        and all(label_path(year, h).is_file() for h in horizons)
    )


def build_season_stack(
    year: int,
    *,
    cube,
    fire_ds,
    anomaly,
    all_years: list[int],
    features: list[str],
    horizons: list[int],
) -> dict:
    """Write one season's dense feature stack and label rasters."""
    h, w = grid.shape()
    dates = ftable._as_dates(cube["ndvi"].sel(time=str(year))["time"].values)
    fire = fire_ds["fire"].sel(time=str(year)).values.astype(np.uint8)
    history = ftable.replay_history(fire_ds, [y for y in all_years if y < year])

    out = np.lib.format.open_memmap(
        feature_path(year),
        mode="w+",
        dtype=FLOAT_DTYPE,
        shape=(len(dates), len(features), h, w),
    )
    index = {name: i for i, name in enumerate(features)}
    filled = np.zeros(len(features), dtype=bool)

    for name, values in ftable.iter_year_features(
        cube, year, dates, fire, anomaly=anomaly, history=history
    ):
        col = index.get(name)
        if col is None:
            continue
        block = np.asarray(values, dtype=np.float32)
        if block.ndim == 2:
            block = np.broadcast_to(block, (len(dates), h, w))
        # NaN is everything outside the Nepal footprint (~60 % of the raster).
        # Convolutions cannot propagate it, so it is zeroed here and excluded
        # from the loss by the validity mask instead.
        out[:, col] = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0).astype(
            FLOAT_DTYPE
        )
        filled[col] = True

    missing = [features[i] for i in np.where(~filled)[0]]
    if missing:
        raise RuntimeError(f"features not produced for {year}: {missing}")
    out.flush()
    del out

    label_sets = derived.horizon_labels(fire, horizons)
    valid = {}
    for horizon in horizons:
        np.save(
            label_path(year, horizon),
            label_sets[f"label_h{horizon}"].astype(np.uint8),
        )
        valid[str(horizon)] = label_sets[f"valid_h{horizon}"].tolist()

    meta = {
        "year": year,
        "dates": [d.isoformat() for d in dates],
        "features": features,
        "horizons": horizons,
        "valid_days": valid,
        "shape": [len(dates), len(features), h, w],
    }
    meta_path(year).write_text(json.dumps(meta), encoding="utf-8")
    return meta


def build_all(
    years: list[int] | None = None,
    *,
    horizons: list[int] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> list[int]:
    settings = load_settings()
    all_years = sorted(settings.years.all)
    years = years or all_years
    horizons = horizons or [1, 7]
    features = ftable.feature_names()

    cube = ftable.open_cube()
    fire_ds = ftable._fire_cube()
    anomaly = ftable.build_anomaly(cube, all_years)

    built = []
    for year in years:
        if is_cached(year, horizons) and not force:
            if verbose:
                print(f"  {year}: cached")
            continue
        meta = build_season_stack(
            year, cube=cube, fire_ds=fire_ds, anomaly=anomaly, all_years=all_years,
            features=features, horizons=horizons,
        )
        built.append(year)
        if verbose:
            size = feature_path(year).stat().st_size / 1e9
            print(f"  {year}: {tuple(meta['shape'])} · {size:.2f} GB")
    return built


class SeasonStack:
    """Read-only view of one cached season."""

    def __init__(self, year: int, horizon: int = 1):
        self.year = year
        self.horizon = horizon
        self.meta = json.loads(meta_path(year).read_text(encoding="utf-8"))
        self.features: list[str] = self.meta["features"]
        self.dates = [np.datetime64(d) for d in self.meta["dates"]]
        self.features_memmap = np.load(feature_path(year), mmap_mode="r")
        self.labels = np.load(label_path(year, horizon), mmap_mode="r")
        self.valid_days = np.asarray(self.meta["valid_days"][str(horizon)], dtype=bool)

    @property
    def n_days(self) -> int:
        return self.features_memmap.shape[0]

    def day(self, t: int) -> np.ndarray:
        """One (C, H, W) plane as float32 — a contiguous read."""
        return np.asarray(self.features_memmap[t], dtype=np.float32)

    def label_day(self, t: int) -> np.ndarray:
        return np.asarray(self.labels[t], dtype=np.float32)


__all__ = [
    "SeasonStack",
    "build_all",
    "build_season_stack",
    "feature_path",
    "is_cached",
    "label_path",
    "meta_path",
    "stacks_dir",
]
