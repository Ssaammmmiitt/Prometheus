from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import APIRouter, Query

from prometheus import grid
from prometheus.features import forest
from prometheus.features import table as ftable
from prometheus.models.predict import _as_date

router_app = APIRouter()


@lru_cache(maxsize=1)
def _fire_cube():
    return ftable._fire_cube()


def _cell_centroids(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) arrays for given mask indices."""
    from rasterio.transform import xy

    xs = xy(grid.transform(), rows, cols, offset="center")[0]
    ys = xy(grid.transform(), rows, cols, offset="center")[1]
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _feature(lon: float, lat: float, *, when: str) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "properties": {"date": when},
    }


def router() -> APIRouter:
    @router_app.get("/fires/active")
    def active_fires(
        as_of: str = Query(None, description="YYYY-MM-DD; defaults to today"),
        lookback_days: int = Query(2, ge=1, le=7),
        limit: int = Query(500, ge=10, le=5000),
    ):
        day = _as_date(as_of) if as_of else date.today()
        cube = _fire_cube()
        mask = forest.forest_mask()

        # Collect detections in the last `lookback_days` days (inclusive).
        feats: list[dict[str, Any]] = []
        for i in range(lookback_days):
            d = day - timedelta(days=i)
            # FIRMS label is daily binary per cell.
            try:
                fire_day = (
                    cube["fire"]
                    .sel(time=str(d))
                    .values.astype(np.uint8)
                )
            except KeyError:
                continue
            sel = np.where((fire_day > 0) & mask)
            if sel[0].size == 0:
                continue

            # Downsample in case the day is very busy.
            idx = np.arange(sel[0].size)
            if sel[0].size > limit:
                idx = np.random.default_rng(0).choice(idx, size=limit, replace=False)
            rr = sel[0][idx]
            cc = sel[1][idx]
            lon, lat = _cell_centroids(rr, cc)
            feats.extend(_feature(lon[j], lat[j], when=d.isoformat()) for j in range(len(rr)))
            if len(feats) >= limit:
                feats = feats[:limit]
                break

        return {"type": "FeatureCollection", "features": feats}

    return router_app


__all__ = ["router"]

