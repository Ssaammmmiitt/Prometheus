"""Per-district mean / max risk on the forest mask."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features

from prometheus import grid
from prometheus.config import load_settings
from prometheus.features import forest
from prometheus.models.calibrate import classify


def districts_source() -> Path:
    """OSM free Geofabrik extract — admin_level 6 is the 77 districts."""
    return (
        load_settings().paths.resolve("raw")
        / "osm"
        / "nepal-free"
        / "gis_osm_adminareas_a_free_1.shp"
    )


def districts_cache() -> Path:
    return load_settings().paths.resolve("static") / "districts_77.geojson"


@lru_cache(maxsize=1)
def load_districts() -> gpd.GeoDataFrame:
    """
    77 district polygons, cleaned once and cached beside the static layers.

    Names stay as OSM provides them (mixed Devanagari / Latin); `district_id`
    is a stable 1..77 integer after sorting by name.
    """
    cache = districts_cache()
    if cache.is_file():
        gdf = gpd.read_file(cache)
        return gdf.set_crs(grid.crs()) if gdf.crs is None else gdf.to_crs(grid.crs())

    source = districts_source()
    if not source.is_file():
        raise FileNotFoundError(
            f"OSM admin polygons missing at {source}. "
            "Unzip nepal-latest-free.shp.zip into data/raw/osm/nepal-free."
        )
    raw = gpd.read_file(source)
    districts = raw[raw["fclass"] == "admin_level6"].copy()
    if len(districts) != 77:
        raise RuntimeError(f"expected 77 districts, found {len(districts)}")
    districts = districts.sort_values("name").reset_index(drop=True)
    districts["district_id"] = np.arange(1, len(districts) + 1, dtype=np.int16)
    districts["name"] = districts["name"].fillna("unknown").astype(str)
    districts = districts[["district_id", "name", "geometry"]].to_crs(grid.crs())
    # Desktop display doesn't need sub-metre vertices; this cuts the file size
    # by ~5–10× so daily backfills stay manageable.
    districts["geometry"] = districts.geometry.simplify(0.001, preserve_topology=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    districts.to_file(cache, driver="GeoJSON")
    return districts


@lru_cache(maxsize=1)
def district_id_raster() -> np.ndarray:
    """(H, W) int16 with district_id on each cell that falls in a district."""
    districts = load_districts()
    shapes = (
        (geom, int(did))
        for geom, did in zip(districts.geometry, districts["district_id"])
    )
    return rasterio.features.rasterize(
        shapes,
        out_shape=grid.shape(),
        transform=grid.transform(),
        fill=0,
        dtype=np.int16,
        all_touched=False,
    )


def zonal_risk(
    risk_by_horizon: dict[int, np.ndarray],
    *,
    bundle_class_names: list[str] | None = None,
    thresholds_h1: list[float] | None = None,
) -> gpd.GeoDataFrame:
    districts = load_districts().copy()
    codes = district_id_raster()
    mask = forest.forest_mask()
    primary = min(risk_by_horizon)

    mean_primary: list[float] = []
    max_primary: list[float] = []
    n_forest: list[int] = []
    class_idx: list[int] = []
    extras: dict[str, list[float]] = {
        f"{stat}_h{h}": []
        for h in risk_by_horizon
        for stat in ("mean", "max")
    }

    for did in districts["district_id"].to_numpy():
        cell = mask & (codes == int(did))
        n = int(cell.sum())
        n_forest.append(n)
        for h, risk in risk_by_horizon.items():
            vals = risk[cell]
            vals = vals[np.isfinite(vals)]
            mean_v = float(vals.mean()) if vals.size else float("nan")
            max_v = float(vals.max()) if vals.size else float("nan")
            extras[f"mean_h{h}"].append(mean_v)
            extras[f"max_h{h}"].append(max_v)
            if h == primary:
                mean_primary.append(mean_v)
                max_primary.append(max_v)
        if thresholds_h1 is not None and np.isfinite(mean_primary[-1]):
            class_idx.append(int(classify(np.array([mean_primary[-1]]), thresholds_h1)[0]))
        else:
            class_idx.append(-1)

    for key, values in extras.items():
        districts[key] = values
    districts["n_forest_cells"] = n_forest
    districts["mean_risk"] = mean_primary
    districts["max_risk"] = max_primary
    districts["risk_class"] = class_idx
    if bundle_class_names:
        names = {i: n for i, n in enumerate(bundle_class_names)}
        districts["risk_class_name"] = [
            names.get(i, "None") if i >= 0 else "None" for i in class_idx
        ]
    return districts


def write_districts(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    out.to_file(path, driver="GeoJSON")
    return path


__all__ = [
    "district_id_raster",
    "districts_cache",
    "load_districts",
    "write_districts",
    "zonal_risk",
]
