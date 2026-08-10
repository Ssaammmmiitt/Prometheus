"""Assemble the tabular training set from the feature cube and fire labels."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from prometheus import grid
from prometheus.config import load_settings
from prometheus.features import derived, forest
from prometheus.features.cube import cube_path, open_cube

# Columns that describe a row rather than predict with it.
META_COLUMNS = ("year", "month", "doy", "row", "col", "lat", "lon")
LABEL_PREFIX = "label_h"

STATS_VERSION = 1


def feature_names() -> list[str]:
    return list(load_settings().features.all_names)


def train_table_path() -> Path:
    return load_settings().paths.resolve("cube") / "train_table.parquet"


def norm_stats_path(version: int = STATS_VERSION) -> Path:
    return load_settings().paths.resolve("models") / f"norm_stats_v{version}.json"


def _fire_cube():
    import xarray as xr

    path = load_settings().paths.resolve("cube") / "fire_daily.zarr"
    if not path.exists():
        raise FileNotFoundError(f"Fire labels missing: {path}")
    return xr.open_zarr(path, consolidated=False)


def _as_dates(values: np.ndarray) -> list[date]:
    return [pd.Timestamp(v).date() for v in values]


def _sample_year(
    labels: np.ndarray,
    valid_days: np.ndarray,
    mask: np.ndarray,
    dates: list[date],
    ratio: int,
    rng: np.random.Generator,
    pos_budget: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick every positive cell-day (up to a budget) plus stratified negatives.

    Negatives are drawn per calendar month so the sample keeps the season's
    shape instead of collapsing onto April, when most fires happen.
    """
    months = np.array([d.month for d in dates])
    t_idx, r_idx, c_idx = [], [], []

    for month in np.unique(months):
        days = np.where((months == month) & valid_days)[0]
        if not len(days):
            continue
        block = labels[days]
        pos_t, pos_r, pos_c = np.where(block & mask[None, ...])
        n_pos = len(pos_t)
        if n_pos == 0:
            continue

        if pos_budget is not None and n_pos > pos_budget:
            keep = rng.choice(n_pos, size=pos_budget, replace=False)
            pos_t, pos_r, pos_c = pos_t[keep], pos_r[keep], pos_c[keep]
            n_pos = pos_budget

        neg_pool_t, neg_pool_r, neg_pool_c = np.where(~block & mask[None, ...])
        n_neg = min(n_pos * ratio, len(neg_pool_t))
        pick = rng.choice(len(neg_pool_t), size=n_neg, replace=False)

        t_idx.append(np.concatenate([days[pos_t], days[neg_pool_t[pick]]]))
        r_idx.append(np.concatenate([pos_r, neg_pool_r[pick]]))
        c_idx.append(np.concatenate([pos_c, neg_pool_c[pick]]))

    if not t_idx:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    return (
        np.concatenate(t_idx),
        np.concatenate(r_idx),
        np.concatenate(c_idx),
    )


def iter_year_features(
    cube,
    year: int,
    dates: list[date],
    fire: np.ndarray,
    *,
    anomaly: derived.SeasonAnomaly,
    history: derived.FireHistory,
):
    """
    Yield every predictor for one season, one array at a time.

    Dynamic features come back as (T, H, W) and static ones as (H, W). Streaming
    them keeps peak memory at a single array, and lets the sampler and the
    full-grid scorer share one definition of what a feature is — the two must
    agree exactly or training and inference would silently diverge.
    """
    settings = load_settings()
    year_cube = cube.sel(time=str(year))

    direct = settings.features.weather_daily + ["ndvi", "evi"] + settings.features.thermal
    rolling_sources = {src for src, _, _ in derived.ROLLING_WINDOWS.values()}
    cached: dict[str, np.ndarray] = {}

    for name in direct:
        if name not in year_cube:
            continue
        block = year_cube[name].values.astype(np.float32)
        if name in rolling_sources or name == "ndvi":
            cached[name] = block
        yield name, block

    for name, values in derived.rolling_weather(cached).items():
        yield name, values

    if "precip" in cached:
        cdd, dsr = derived.dry_spell(cached["precip"])
        yield "consecutive_dry_days", cdd
        yield "days_since_rain", dsr

    if "ndvi" in cached:
        yield "ndvi_anomaly", anomaly.anomaly(year, dates, cached["ndvi"])
    cached.clear()

    for name, values in history.process_year(dates, fire).items():
        yield name, values
    yield "fire_clim", derived.fire_climatology_slice(dates)

    static_names = (
        settings.features.terrain_static
        + settings.features.landcover_static
        + settings.features.human_static
    )
    for name in static_names:
        if name in cube:
            yield name, cube[name].values.astype(np.float32)

    sin, cos = derived.day_of_year_encoding(dates)
    h, w = grid.shape()
    yield "doy_sin", np.broadcast_to(sin[:, None, None], (len(dates), h, w))
    yield "doy_cos", np.broadcast_to(cos[:, None, None], (len(dates), h, w))


def build_year_rows(
    cube,
    fire_ds,
    year: int,
    *,
    anomaly: derived.SeasonAnomaly,
    history: derived.FireHistory,
    ratio: int,
    rng: np.random.Generator,
    pos_budget: int | None,
    horizons: list[int],
) -> pd.DataFrame:
    """One season's worth of sampled rows, features already computed."""
    year_cube = cube.sel(time=str(year))
    dates = _as_dates(year_cube["time"].values)
    mask = forest.forest_mask()

    fire = fire_ds["fire"].sel(time=str(year)).values.astype(np.uint8)
    label_sets = derived.horizon_labels(fire, horizons)

    primary = horizons[0]
    labels = label_sets[f"label_h{primary}"]
    valid_days = label_sets[f"valid_h{primary}"]

    t_idx, r_idx, c_idx = _sample_year(
        labels, valid_days, mask, dates, ratio, rng, pos_budget
    )
    if not len(t_idx):
        return pd.DataFrame()

    columns: dict[str, np.ndarray] = {}
    for name, values in iter_year_features(
        cube, year, dates, fire, anomaly=anomaly, history=history
    ):
        if values.ndim == 3:
            columns[name] = np.asarray(values[t_idx, r_idx, c_idx], dtype=np.float32)
        else:
            columns[name] = np.asarray(values[r_idx, c_idx], dtype=np.float32)

    # ---- labels and metadata ----
    for h in horizons:
        columns[f"{LABEL_PREFIX}{h}"] = label_sets[f"label_h{h}"][
            t_idx, r_idx, c_idx
        ].astype(np.uint8)

    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=np.int16)
    month = np.array([d.month for d in dates], dtype=np.int8)
    columns["year"] = np.full(len(t_idx), year, dtype=np.int16)
    columns["month"] = month[t_idx]
    columns["doy"] = doy[t_idx]
    columns["row"] = r_idx.astype(np.int16)
    columns["col"] = c_idx.astype(np.int16)
    columns["lat"] = cube["y"].values[r_idx].astype(np.float32)
    columns["lon"] = cube["x"].values[c_idx].astype(np.float32)

    df = pd.DataFrame(columns)
    for col in df.columns:
        if col.startswith(LABEL_PREFIX) or col in META_COLUMNS:
            continue
        df[col] = df[col].astype(np.float32)
    return df


def build_train_table(
    years: list[int] | None = None,
    *,
    ratio: int | None = None,
    positive_cap: int | None = 100_000,
    out_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Build train_table.parquet: every sampled cell-day with its predictors.

    Positives can be capped for tractability. That only thins the training
    sample; evaluation still scores every forest cell straight off the cube, so
    the reported metrics are unaffected by this choice.
    """
    settings = load_settings()
    years = sorted(years or settings.years.all)
    ratio = ratio or settings.modeling.positive_negative_ratio
    out_path = out_path or train_table_path()
    horizon_list = sorted(settings.modeling.horizons)

    if not cube_path().exists():
        raise FileNotFoundError("features_daily.zarr missing — run build_feature_cube.py")

    cube = open_cube()
    fire_ds = _fire_cube()
    rng = np.random.default_rng(settings.modeling.random_seed)

    if verbose:
        print(f"years {years[0]}-{years[-1]} · ratio 1:{ratio} · horizons {horizon_list}")

    # Pass 1: NDVI totals for the leave-one-year-out anomaly climatology.
    anomaly = derived.SeasonAnomaly()
    for year in years:
        block = cube["ndvi"].sel(time=str(year))
        anomaly.add_year(year, _as_dates(block["time"].values), block.values.astype(np.float32))
    if verbose:
        print("ndvi climatology accumulated")

    # The cap is spread evenly over every year-month stratum, so no single
    # April dominates the sample.
    n_strata = len(years) * len(settings.season.months)
    per_stratum_budget = None if not positive_cap else max(positive_cap // n_strata, 1)

    history = derived.FireHistory(grid.shape())
    frames: list[pd.DataFrame] = []
    for year in years:
        df = build_year_rows(
            cube,
            fire_ds,
            year,
            anomaly=anomaly,
            history=history,
            ratio=ratio,
            rng=rng,
            pos_budget=per_stratum_budget,
            horizons=horizon_list,
        )
        frames.append(df)
        if verbose:
            pos = int(df[f"{LABEL_PREFIX}{horizon_list[0]}"].sum()) if len(df) else 0
            print(f"  {year}: {len(df):>9,} rows · {pos:>7,} positives", flush=True)

    table = pd.concat(frames, ignore_index=True)
    del frames

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False, compression="zstd")

    stats = compute_norm_stats(table, years)
    stats_path = write_norm_stats(stats)

    primary = f"{LABEL_PREFIX}{horizon_list[0]}"
    report = {
        "path": str(out_path),
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
        "feature_columns": len(present_features(table)),
        "positives": int(table[primary].sum()),
        "positive_rate": float(table[primary].mean()),
        "ratio": ratio,
        "years": years,
        "size_bytes": out_path.stat().st_size,
        "norm_stats_path": str(stats_path),
        "missing_features": sorted(set(feature_names()) - set(table.columns)),
    }
    report_path = out_path.parent / "train_table_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def present_features(table: pd.DataFrame) -> list[str]:
    return [c for c in feature_names() if c in table.columns]


def replay_history(fire_ds, years: list[int]) -> derived.FireHistory:
    """Advance fire-history state through past seasons without keeping outputs."""
    history = derived.FireHistory(grid.shape())
    for year in sorted(years):
        block = fire_ds["fire"].sel(time=str(year))
        history.advance(_as_dates(block["time"].values), block.values.astype(np.uint8))
    return history


def build_anomaly(cube, years: list[int]) -> derived.SeasonAnomaly:
    """NDVI totals for the leave-one-year-out anomaly climatology."""
    anomaly = derived.SeasonAnomaly()
    for year in sorted(years):
        block = cube["ndvi"].sel(time=str(year))
        anomaly.add_year(
            year, _as_dates(block["time"].values), block.values.astype(np.float32)
        )
    return anomaly


def year_grid_features(
    year: int,
    *,
    cube=None,
    fire_ds=None,
    anomaly: derived.SeasonAnomaly | None = None,
    all_years: list[int] | None = None,
    features: list[str] | None = None,
) -> dict:
    """
    Materialise every predictor for every forest cell of one season.

    This is what makes the model comparable to the baselines: they are scored on
    the whole grid, so the model has to be too. Sampling belongs to training
    only. Returns the design matrix as (n_days * n_cells, n_features).
    """
    settings = load_settings()
    all_years = sorted(all_years or settings.years.all)
    cube = cube if cube is not None else open_cube()
    fire_ds = fire_ds if fire_ds is not None else _fire_cube()
    anomaly = anomaly if anomaly is not None else build_anomaly(cube, all_years)
    features = features or feature_names()

    dates = _as_dates(cube["ndvi"].sel(time=str(year))["time"].values)
    fire = fire_ds["fire"].sel(time=str(year)).values.astype(np.uint8)

    mask = forest.forest_mask()
    rows, cols = np.where(mask)
    n_days, n_cells = len(dates), len(rows)

    history = replay_history(fire_ds, [y for y in all_years if y < year])
    index = {name: i for i, name in enumerate(features)}
    matrix = np.empty((n_days * n_cells, len(features)), dtype=np.float32)
    filled = np.zeros(len(features), dtype=bool)

    for name, values in iter_year_features(
        cube, year, dates, fire, anomaly=anomaly, history=history
    ):
        col = index.get(name)
        if col is None:
            continue
        if values.ndim == 3:
            matrix[:, col] = np.asarray(values[:, rows, cols], dtype=np.float32).ravel()
        else:
            matrix[:, col] = np.tile(
                np.asarray(values[rows, cols], dtype=np.float32), n_days
            )
        filled[col] = True

    missing = [features[i] for i in np.where(~filled)[0]]
    if missing:
        raise RuntimeError(f"features not produced for {year}: {missing}")

    return {
        "matrix": matrix,
        "features": features,
        "dates": dates,
        "rows": rows,
        "cols": cols,
        "n_days": n_days,
        "n_cells": n_cells,
        "fire": fire,
    }


def compute_norm_stats(table: pd.DataFrame, years: list[int]) -> dict:
    """
    Per-fold normalisation statistics.

    Each leave-one-year-out fold gets stats from its own training years only;
    using whole-dataset statistics would leak the held-out season's
    distribution into the model that is supposed to be blind to it.
    """
    features = present_features(table)

    def describe(frame: pd.DataFrame) -> dict:
        out = {}
        for name in features:
            col = frame[name].to_numpy(dtype=np.float64)
            col = col[np.isfinite(col)]
            if not col.size:
                out[name] = {"mean": 0.0, "std": 1.0, "p1": 0.0, "p99": 0.0}
                continue
            std = float(col.std())
            out[name] = {
                "mean": float(col.mean()),
                "std": std if std > 1e-9 else 1.0,
                "p1": float(np.percentile(col, 1)),
                "p99": float(np.percentile(col, 99)),
            }
        return out

    folds = {
        str(held_out): describe(table[table["year"] != held_out]) for held_out in years
    }
    return {
        "version": STATS_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "scheme": "leave_one_year_out",
        "features": features,
        "folds": folds,
    }


def write_norm_stats(stats: dict) -> Path:
    path = norm_stats_path(stats.get("version", STATS_VERSION))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


def load_train_table(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or train_table_path())


__all__ = [
    "LABEL_PREFIX",
    "META_COLUMNS",
    "STATS_VERSION",
    "build_train_table",
    "build_year_rows",
    "compute_norm_stats",
    "feature_names",
    "load_train_table",
    "norm_stats_path",
    "present_features",
    "train_table_path",
    "write_norm_stats",
]
