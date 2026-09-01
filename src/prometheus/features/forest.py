"""Static layers on the canonical grid: terrain, land cover, forest mask, human."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import rasterio

from prometheus import grid
from prometheus.config import load_settings
from prometheus.features.warp import gee_static_dir, read_to_grid

# ESA WorldCover v200 classes we treat as burnable vegetation.
BURNABLE = ("tree", "shrub", "grass")
# Minimum burnable fraction for a 1 km cell to enter the forest mask.
# Raising this to 0.5 drops ~9% of cells but also ~3% of the fire pixel-days we
# are trying to predict, so the looser threshold is the better trade here.
FOREST_FRACTION_MIN = 0.25
# Nepal's treeline tops out around 4000-4500 m; above it there is no continuous
# fuel. Cutting there removes ~3,800 cells and costs no positives at all.
ELEVATION_MAX_M = 4500.0

WORLDCOVER_BANDS = (
    "tree",
    "shrub",
    "grass",
    "crop",
    "built",
    "bare",
    "snow",
    "water",
    "wetland",
    "mangrove",
    "moss",
)


@lru_cache(maxsize=1)
def worldcover_fractions() -> dict[str, np.ndarray]:
    path = gee_static_dir() / "worldcover_frac.tif"
    if not path.is_file():
        raise FileNotFoundError(f"WorldCover fractions missing: {path}")
    data, names = read_to_grid(path, resampling="average")
    return {n: data[i] for i, n in enumerate(names)}


@lru_cache(maxsize=1)
def terrain() -> dict[str, np.ndarray]:
    """elevation, slope, aspect_sin, aspect_cos, twi on the canonical grid."""
    out: dict[str, np.ndarray] = {}
    esa = gee_static_dir() / "elev_slope_aspect.tif"
    if esa.is_file():
        data, names = read_to_grid(esa, resampling="bilinear")
        lookup = {n: data[i] for i, n in enumerate(names)}
        out["elevation"] = lookup["elev"]
        out["slope"] = lookup["slope"]
        aspect = np.deg2rad(lookup["aspect"])
        out["aspect_sin"] = np.sin(aspect).astype(np.float32)
        out["aspect_cos"] = np.cos(aspect).astype(np.float32)
    else:
        settings = load_settings()
        with rasterio.open(settings.paths.resolve("elevation")) as src:
            out["elevation"] = src.read(1).astype(np.float32)
        with rasterio.open(settings.paths.resolve("slope")) as src:
            out["slope"] = src.read(1).astype(np.float32)
        out["aspect_sin"] = np.zeros(grid.shape(), dtype=np.float32)
        out["aspect_cos"] = np.zeros(grid.shape(), dtype=np.float32)

    twi_path = gee_static_dir() / "twi.tif"
    if twi_path.is_file():
        twi, _ = read_to_grid(twi_path, resampling="bilinear")
        out["twi"] = twi[0]
    return out


@lru_cache(maxsize=1)
def human_static() -> dict[str, np.ndarray]:
    """Distance-to-road / settlement and physiographic region codes."""
    static_dir = load_settings().paths.resolve("static")
    out: dict[str, np.ndarray] = {}
    for name, key in (
        ("dist_road", "dist_road.tif"),
        ("dist_settlement", "dist_settlement.tif"),
        ("physio_region", "physio_regions.tif"),
    ):
        path = static_dir / key
        if not path.is_file():
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
        out[name] = arr
    return out


@lru_cache(maxsize=1)
def forest_mask() -> np.ndarray:
    """
    Burnable-vegetation cells inside Nepal.

    Keep tree cover, shrubland and grassland; drop water, snow/ice, bare rock,
    built-up and cropland-dominated cells. Modelling only these cells keeps the
    negative class meaningful instead of padding it with cells that cannot burn.
    """
    mask = (burnable_fraction() >= FOREST_FRACTION_MIN) & grid.nepal_mask()
    elevation = terrain().get("elevation")
    if elevation is not None:
        mask &= np.nan_to_num(elevation, nan=0.0) <= ELEVATION_MAX_M
    return mask


def burnable_fraction() -> np.ndarray:
    fr = worldcover_fractions()
    total = np.zeros(grid.shape(), dtype=np.float32)
    for name in BURNABLE:
        total += np.nan_to_num(fr[name], nan=0.0)
    return total


def static_layers() -> dict[str, np.ndarray]:
    """Every 2D layer that goes into the cube."""
    layers: dict[str, np.ndarray] = {}
    layers.update(terrain())
    fr = worldcover_fractions()
    for name in ("tree", "shrub", "grass", "crop", "built", "bare", "water", "snow"):
        layers[f"{name}_frac"] = fr[name]
    layers.update(human_static())
    layers["forest_mask"] = forest_mask().astype(np.float32)
    layers["nepal_mask"] = grid.nepal_mask().astype(np.float32)
    return layers


def burned_cell_capture() -> float | None:
    """
    Share of ever-burned cells that survive the mask.

    This is the number that decides the threshold: a tighter mask shrinks the
    negative pool a little but throws away positives, which is the expensive
    mistake for a rare-event model.
    """
    import xarray as xr

    from prometheus.features.cube import strip_finder_junk

    path = load_settings().paths.resolve("cube") / "fire_daily.zarr"
    if not path.exists():
        return None
    strip_finder_junk(path)
    ds = xr.open_zarr(path, consolidated=False)
    burned = ds["fire"].values.sum(axis=0) > 0
    if not burned.any():
        return None
    return float((burned & forest_mask()).sum() / burned.sum())


def summary() -> dict:
    mask = grid.nepal_mask()
    forest = forest_mask()
    out = {
        "nepal_cells": int(mask.sum()),
        "forest_cells": int(forest.sum()),
        "forest_share_of_nepal": float(forest.sum() / max(mask.sum(), 1)),
        "burnable_fraction_min": FOREST_FRACTION_MIN,
        "elevation_max_m": ELEVATION_MAX_M,
        "burnable_classes": list(BURNABLE),
    }
    try:
        capture = burned_cell_capture()
    except Exception:
        capture = None
    if capture is not None:
        out["burned_cell_capture"] = capture
    return out


__all__ = [
    "BURNABLE",
    "FOREST_FRACTION_MIN",
    "burnable_fraction",
    "forest_mask",
    "human_static",
    "static_layers",
    "summary",
    "terrain",
    "worldcover_fractions",
]
