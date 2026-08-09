#!/usr/bin/env python3
"""Build local static layers: dist-to-road, dist-to-settlement, physio regions.

All outputs match configs/base.yaml canonical 1 km grid.

Examples
--------
  python scripts/build_local_static.py --osm-dir data/raw/osm/nepal-free
  python scripts/build_local_static.py --osm-dir data/raw/osm/nepal-free \\
      --physio-vector data/raw/osm/nepal_physio.gpkg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features as rio_features
from scipy import ndimage

# project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prometheus.config import load_settings  # noqa: E402
from prometheus.grid import nepal_mask, nodata, profile, shape, transform  # noqa: E402

# OSM free shapefile names (Geofabrik)
ROAD_CANDIDATES = [
    "gis_osm_roads_free_1.shp",
    "gis_osm_roads_free_1.gpkg",
]
PLACE_CANDIDATES = [
    "gis_osm_places_free_1.shp",
    "gis_osm_places_free_1.gpkg",
]
# Settlement-like landuse as fallback
LANDUSE_CANDIDATES = [
    "gis_osm_landuse_free_1.shp",
    "gis_osm_landuse_free_1.gpkg",
]

PHYSIO_CODES = {
    1: "Terai",
    2: "Chure",
    3: "MiddleMountains",
    4: "HighMountains",
}

# Elevation proxy thresholds (meters)
ELEV_TERAI_MAX = 300.0
ELEV_CHURE_MAX = 1000.0
ELEV_MID_MAX = 3000.0


def _find(dir_path: Path, names: list[str]) -> Path | None:
    for n in names:
        p = dir_path / n
        if p.is_file():
            return p
    # recursive one level
    for n in names:
        hits = list(dir_path.rglob(n))
        if hits:
            return hits[0]
    return None


def _rasterize_geoms(geoms, out_shape, burn: int = 1) -> np.ndarray:
    if not geoms:
        return np.zeros(out_shape, dtype=np.uint8)
    return rio_features.rasterize(
        ((g, burn) for g in geoms if g is not None and not g.is_empty),
        out_shape=out_shape,
        transform=transform(),
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


def _distance_km(presence: np.ndarray) -> np.ndarray:
    """Euclidean distance in km from nearest True/1 pixel (~1 km cells)."""
    # distance_transform_edt on zeros: distance to nearest non-zero
    # so presence should be 1 where feature exists
    inv = presence == 0
    px = float(abs(transform().a))  # degrees
    # rough km/degree at Nepal mid-lat ~28°: 1° lat ≈ 111.32 km; lon × cos(lat)
    # Using config pixel size in degrees and isotropic approx at lat 28.4
    lat0 = 28.4
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat0))
    # anisotropic: stretch so EDT spacing matches km
    # Use mean of lon/lat deg→km for simple isotropic dx
    dy_km = abs(transform().e) * km_per_deg_lat
    dx_km = abs(transform().a) * km_per_deg_lon
    sampling = (dy_km, dx_km)

    if not np.any(presence):
        # no features: large constant distance
        return np.full(presence.shape, 999.0, dtype=np.float32)

    dist = ndimage.distance_transform_edt(inv, sampling=sampling)
    return dist.astype(np.float32)


def _write_float(path: Path, data: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = data.astype(np.float32)
    arr = np.where(mask, arr, nodata())
    prof = profile(dtype="float32", count=1)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def _write_int(path: Path, data: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = data.astype(np.int16)
    arr = np.where(mask, arr, -9999)
    prof = profile(dtype="int16", count=1, nodata_value=-9999)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def load_roads(osm_dir: Path):
    import geopandas as gpd

    path = _find(osm_dir, ROAD_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            f"No roads shapefile in {osm_dir}. Expected one of {ROAD_CANDIDATES}"
        )
    print(f"  roads: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return list(gdf.geometry.values)


def load_settlements(osm_dir: Path):
    import geopandas as gpd

    geoms = []
    path = _find(osm_dir, PLACE_CANDIDATES)
    if path is not None:
        print(f"  places: {path}")
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        geoms.extend(list(gdf.geometry.values))

    path_lu = _find(osm_dir, LANDUSE_CANDIDATES)
    if path_lu is not None:
        print(f"  landuse (residential filter): {path_lu}")
        gdf = gpd.read_file(path_lu)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        # fclass or similar
        col = None
        for c in gdf.columns:
            if c.lower() in ("fclass", "class", "landuse", "type"):
                col = c
                break
        if col is not None:
            keys = {"residential", "industrial", "commercial", "retail", "farmyard"}
            gdf = gdf[gdf[col].astype(str).str.lower().isin(keys)]
        geoms.extend(list(gdf.geometry.values))

    if not geoms:
        raise FileNotFoundError(
            f"No places/settlement geometries found under {osm_dir}. "
            f"Need {PLACE_CANDIDATES} or landuse."
        )
    return geoms


def build_distances(osm_dir: Path, out_dir: Path, mask: np.ndarray) -> None:
    print("Rasterizing roads…")
    roads = load_roads(osm_dir)
    road_r = _rasterize_geoms(roads, shape())
    print(f"  road pixels: {int(road_r.sum())}")
    dist_road = _distance_km(road_r)
    _write_float(out_dir / "dist_road.tif", dist_road, mask)
    print(f"  wrote {out_dir / 'dist_road.tif'}")

    print("Rasterizing settlements…")
    settles = load_settlements(osm_dir)
    set_r = _rasterize_geoms(settles, shape())
    print(f"  settlement pixels: {int(set_r.sum())}")
    dist_set = _distance_km(set_r)
    _write_float(out_dir / "dist_settlement.tif", dist_set, mask)
    print(f"  wrote {out_dir / 'dist_settlement.tif'}")


def find_elevation() -> Path | None:
    settings = load_settings()
    candidates = [
        settings.paths.resolve("elevation"),
        Path("data/static/elevation_static_srtm.tif"),
        Path("data/raw/gee/static/elev_slope_aspect.tif"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def physio_from_elevation(mask: np.ndarray) -> np.ndarray:
    elev_path = find_elevation()
    if elev_path is None:
        raise FileNotFoundError(
            "Need elevation GeoTIFF for physio proxy. "
            "Place data/static/elevation_static_srtm.tif or export GEE elev first."
        )
    print(f"  elevation for physio: {elev_path}")
    with rasterio.open(elev_path) as src:
        # take band 1 (elev if multi-band from GEE)
        elev = src.read(1).astype(np.float32)
        if src.shape != shape():
            # warp to canonical grid
            from rasterio.warp import reproject, Resampling

            dest = np.full(shape(), np.nan, dtype=np.float32)
            reproject(
                source=elev,
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs or "EPSG:4326",
                dst_transform=transform(),
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
            elev = dest

    out = np.zeros(shape(), dtype=np.int16)
    out[elev < ELEV_TERAI_MAX] = 1
    out[(elev >= ELEV_TERAI_MAX) & (elev < ELEV_CHURE_MAX)] = 2
    out[(elev >= ELEV_CHURE_MAX) & (elev < ELEV_MID_MAX)] = 3
    out[elev >= ELEV_MID_MAX] = 4
    out[~mask] = 0
    return out


def physio_from_vector(path: Path, mask: np.ndarray) -> np.ndarray:
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    name_col = None
    for c in gdf.columns:
        cl = c.lower()
        if any(k in cl for k in ("name", "region", "class", "physio", "zone")):
            name_col = c
            break
    if name_col is None:
        raise ValueError(f"No region name column in {path}. Columns: {list(gdf.columns)}")

    def code_for(val: str) -> int:
        s = str(val).lower()
        if "terai" in s or "tarai" in s:
            return 1
        if "chure" in s or "siwalik" in s:
            return 2
        if "middle" in s or "mid-hill" in s or "mahāb" in s or "mahab" in s:
            return 3
        if "high" in s or "himal" in s:
            return 4
        return 0

    shapes = []
    for _, row in gdf.iterrows():
        c = code_for(row[name_col])
        if c and row.geometry is not None and not row.geometry.is_empty:
            shapes.append((row.geometry, c))

    arr = rio_features.rasterize(
        shapes,
        out_shape=shape(),
        transform=transform(),
        fill=0,
        dtype=np.int16,
        all_touched=True,
    )
    arr = np.where(mask, arr, 0)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--osm-dir",
        type=Path,
        required=True,
        help="Folder with Geofabrik free shapefiles (gis_osm_roads_free_1.shp …)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: data/static from config",
    )
    ap.add_argument(
        "--physio-vector",
        type=Path,
        default=None,
        help="Optional polygon file with region names; else elevation bands",
    )
    ap.add_argument(
        "--skip-distances",
        action="store_true",
        help="Only build physio regions",
    )
    args = ap.parse_args()

    settings = load_settings()
    out_dir = args.out_dir or settings.paths.resolve("static")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = nepal_mask()
    print(f"Grid {shape()} · valid Nepal pixels {int(mask.sum())}")
    print(f"Output dir: {out_dir}")

    if not args.skip_distances:
        if not args.osm_dir.is_dir():
            raise SystemExit(f"--osm-dir not found: {args.osm_dir}")
        build_distances(args.osm_dir, out_dir, mask)

    print("Physiographic regions…")
    if args.physio_vector is not None:
        phys = physio_from_vector(args.physio_vector, mask)
        method = f"vector:{args.physio_vector}"
    else:
        phys = physio_from_elevation(mask)
        method = (
            f"elevation_proxy: Terai<{ELEV_TERAI_MAX}, "
            f"Chure<{ELEV_CHURE_MAX}, Mid<{ELEV_MID_MAX}, High≥"
        )
    _write_int(out_dir / "physio_regions.tif", phys, mask)
    legend = {
        "codes": PHYSIO_CODES,
        "method": method,
        "note": "0 / nodata outside Nepal mask",
    }
    legend_path = out_dir / "physio_regions.json"
    legend_path.write_text(json.dumps(legend, indent=2))
    print(f"  wrote {out_dir / 'physio_regions.tif'}")
    print(f"  legend {legend_path}")
    for code, name in PHYSIO_CODES.items():
        n = int(((phys == code) & mask).sum())
        print(f"    {code} {name}: {n} cells")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
