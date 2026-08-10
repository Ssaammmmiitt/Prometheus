"""Assemble the aligned daily feature cube (features_daily.zarr)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from prometheus import grid
from prometheus.config import load_settings
from prometheus.features import forest, vegetation, weather

TIME_CHUNK = 32
SPACE_CHUNK = 256
EPOCH = date(1970, 1, 1)

VEG_PRODUCTS = (
    # (folder, prefix, valid range, output variable names)
    ("ndvi", "ndvi", vegetation.NDVI_VALID, ("ndvi", "evi")),
    ("lst", "lst", vegetation.LST_VALID, ("lst_day", "lst_night")),
)


def season_dates(years: list[int]) -> list[date]:
    s = load_settings().season
    out: list[date] = []
    for y in sorted(years):
        d = date(y, s.start_month, s.start_day)
        end = date(y, s.end_month, s.end_day)
        while d <= end:
            out.append(d)
            d += timedelta(days=1)
    return out


def cube_path() -> Path:
    return load_settings().paths.resolve("cube") / "features_daily.zarr"


def dynamic_variables() -> list[str]:
    return list(weather.WEATHER_VARS) + ["ndvi", "evi", "lst_day", "lst_night", "lst_diff"]


def _compressors() -> list[BloscCodec]:
    return [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")]


def _init_store(path: Path, dates: list[date], overwrite: bool) -> zarr.Group:
    import shutil

    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = grid.shape()
    t = grid.transform()
    g = zarr.open_group(path, mode="w")

    time_arr = g.create_array(
        name="time",
        shape=(len(dates),),
        chunks=(len(dates),),
        dtype="int64",
        dimension_names=("time",),
    )
    time_arr[:] = np.array([(d - EPOCH).days for d in dates], dtype=np.int64)
    time_arr.attrs["units"] = "days since 1970-01-01"
    time_arr.attrs["calendar"] = "proleptic_gregorian"

    ys = t.f + (np.arange(h) + 0.5) * t.e
    xs = t.c + (np.arange(w) + 0.5) * t.a
    for name, values, dim in (("y", ys, "y"), ("x", xs, "x")):
        arr = g.create_array(
            name=name,
            shape=values.shape,
            chunks=values.shape,
            dtype="float64",
            dimension_names=(dim,),
        )
        arr[:] = values

    for var in dynamic_variables():
        g.create_array(
            name=var,
            shape=(len(dates), h, w),
            chunks=(TIME_CHUNK, SPACE_CHUNK, SPACE_CHUNK),
            dtype="float16",
            dimension_names=("time", "y", "x"),
            compressors=_compressors(),
            fill_value=np.float16(np.nan),
        )

    for name, layer in forest.static_layers().items():
        arr = g.create_array(
            name=name,
            shape=(h, w),
            chunks=(SPACE_CHUNK, SPACE_CHUNK),
            dtype="float32",
            dimension_names=("y", "x"),
            compressors=_compressors(),
        )
        arr[:] = layer.astype(np.float32)

    g.attrs.update(
        {
            "title": "Prometheus daily wildfire feature cube",
            "crs": load_settings().grid.crs,
            "transform": [float(v) for v in t[:6]],
            "height": h,
            "width": w,
            "season_months": load_settings().season.months,
            "lapse_rate_t": weather.LAPSE_T,
            "lapse_rate_td": weather.LAPSE_TD,
            "surface_pressure_units": "hPa",
            "note": "values are NaN outside the Nepal mask",
        }
    )
    return g


class _NanTracker:
    """Count NaNs inside the forest mask while data streams past."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.cells = int(mask.sum())
        self.nan: dict[str, int] = {}
        self.total: dict[str, int] = {}

    def add(self, var: str, block: np.ndarray) -> None:
        sub = block[:, self.mask]
        self.nan[var] = self.nan.get(var, 0) + int((~np.isfinite(sub)).sum())
        self.total[var] = self.total.get(var, 0) + int(sub.size)

    def fractions(self) -> dict[str, float]:
        return {
            var: (self.nan[var] / self.total[var] if self.total.get(var) else 0.0)
            for var in self.nan
        }


def _write(group: zarr.Group, var: str, start: int, block: np.ndarray) -> None:
    group[var][start : start + block.shape[0]] = block.astype(np.float16)


def build_feature_cube(
    years: list[int] | None = None,
    *,
    overwrite: bool = True,
    path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Downscale, interpolate, and write every layer onto one grid."""
    settings = load_settings()
    years = years or list(settings.years.all)
    dates = season_dates(years)
    path = path or cube_path()

    nepal = grid.nepal_mask()
    forest_mask = forest.forest_mask()
    tracker = _NanTracker(forest_mask)

    if verbose:
        info = forest.summary()
        print(f"grid {grid.shape()} · Nepal {info['nepal_cells']:,} cells")
        print(
            f"forest mask {info['forest_cells']:,} cells "
            f"({info['forest_share_of_nepal']:.1%} of Nepal)"
        )
        print(f"time {len(dates)} days · {years[0]}–{years[-1]}")

    group = _init_store(path, dates, overwrite)
    offsets = {d: i for i, d in enumerate(dates)}

    # Per-pixel composite means, used only where a season has no valid data.
    climatology = {
        folder: vegetation.pixel_climatology(folder, prefix, years, valid)
        for folder, prefix, valid, _ in VEG_PRODUCTS
    }

    for year in years:
        if verbose:
            print(f"\n{year}")
        year_dates = [d for d in dates if d.year == year]
        base = offsets[year_dates[0]]

        # ---- weather: monthly ERA5 stacks, 9 km → 1 km ----
        for month in settings.season.months:
            mdates, fields = weather.downscale_month(year, month)
            keep = [i for i, d in enumerate(mdates) if d in offsets]
            if not keep:
                continue
            start = offsets[mdates[keep[0]]]
            for var in weather.WEATHER_VARS:
                if var not in fields:
                    continue
                block = fields[var][keep]
                block = np.where(nepal[None, ...], block, np.nan)
                tracker.add(var, block)
                _write(group, var, start, block)
            if verbose:
                print(f"  era5 {year}-{month:02d}: {len(keep)} days", flush=True)

        # ---- vegetation / thermal: composites → daily ----
        for folder, prefix, valid, out_names in VEG_PRODUCTS:
            daily, _ = vegetation.daily_year(
                folder, prefix, year, year_dates, valid, climatology=climatology[folder]
            )
            for b, var in enumerate(out_names):
                block = np.where(nepal[None, ...], daily[:, b], np.nan)
                tracker.add(var, block)
                _write(group, var, base, block)
            if folder == "lst":
                diff = np.where(nepal[None, ...], daily[:, 0] - daily[:, 1], np.nan)
                tracker.add("lst_diff", diff)
                _write(group, "lst_diff", base, diff)
            if verbose:
                print(f"  {folder}: {len(year_dates)} daily fields", flush=True)

    fractions = tracker.fractions()
    group.attrs["nan_fraction_in_forest_mask"] = {k: round(v, 6) for k, v in fractions.items()}
    group.attrs["years"] = list(years)

    report = {
        "path": str(path),
        "n_times": len(dates),
        "shape": [len(dates), *grid.shape()],
        "years": list(years),
        "variables": dynamic_variables(),
        "static_variables": sorted(forest.static_layers().keys()),
        "forest": forest.summary(),
        "nan_fraction_in_forest_mask": fractions,
        "max_nan_fraction": max(fractions.values()) if fractions else 0.0,
        "passes_nan_check": all(v <= 0.05 for v in fractions.values()),
        "size_bytes": _dir_size(path),
    }
    report_path = path.parent / "features_daily_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def open_cube(path: Path | None = None):
    import xarray as xr

    return xr.open_zarr(path or cube_path(), consolidated=False)


__all__ = [
    "SPACE_CHUNK",
    "TIME_CHUNK",
    "build_feature_cube",
    "cube_path",
    "dynamic_variables",
    "open_cube",
    "season_dates",
]
