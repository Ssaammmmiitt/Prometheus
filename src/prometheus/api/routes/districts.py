from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from prometheus.infer import io_cog
from prometheus.models.calibrate import classify
from prometheus.models.predict import RiskPredictor

router_app = APIRouter()


@lru_cache(maxsize=1)
def _predictor() -> RiskPredictor:
    return RiskPredictor("latest")


def _district_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ts_cache_path(root: Path) -> Path:
    return Path(root) / "_district_ts.json"


def load_timeseries_table(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Compact per-district daily stats. Built once from the GeoJSON backfill."""
    root = Path(root)
    cache = _ts_cache_path(root)
    files = sorted(root.glob("districts_*.geojson"))
    if cache.is_file() and files:
        newest = max(p.stat().st_mtime for p in files)
        if cache.stat().st_mtime >= newest - 1:
            return json.loads(cache.read_text(encoding="utf-8"))

    table: dict[str, list[dict[str, Any]]] = {}
    for path in files:
        day = path.name.split("_")[1].split(".")[0]
        geo = _district_json(path)
        for feat in geo.get("features", []):
            props = feat.get("properties", {})
            did = str(int(props.get("district_id", -1)))
            table.setdefault(did, []).append(
                {
                    "date": day,
                    "mean_h1": props.get("mean_h1"),
                    "max_h1": props.get("max_h1"),
                    "mean_h7": props.get("mean_h7"),
                    "max_h7": props.get("max_h7"),
                }
            )
    cache.write_text(json.dumps(table), encoding="utf-8")
    return table


def _reclassify_features(
    geojson: dict[str, Any],
    *,
    horizon: int,
    thresholds: list[float],
    class_names: list[str],
):
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    for feat in geojson["features"]:
        props = feat.get("properties", {})
        # Inputs we store from Day 13.
        key = f"mean_h{horizon}"
        prob = props.get(key)
        if prob is None:
            continue
        idx = int(classify(np.asarray([prob], dtype=np.float64), thresholds_arr)[0])
        props["risk_class"] = idx
        props["risk_class_name"] = class_names[idx] if idx >= 0 else "None"
    return geojson


def router_factory(root_fn):
    @router_app.get("/districts")
    def districts(
        date: str,
        horizon: int = Query(1, ge=1, le=7),
    ):
        root = root_fn()
        path = io_cog.districts_path(date, root=root)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"missing {path.name}")

        geo = _district_json(path)
        pred = _predictor()
        thresholds = pred.bundle.horizons[horizon].risk_thresholds
        return _reclassify_features(
            geo,
            horizon=horizon,
            thresholds=thresholds,
            class_names=pred.class_names(),
        )

    @router_app.get("/districts/{district_id}/timeseries")
    def timeseries(
        district_id: int,
        horizon: int = Query(1, ge=1, le=7),
        start: str | None = None,
        end: str | None = None,
    ):
        root = Path(root_fn())
        table = load_timeseries_table(root)
        rows = table.get(str(int(district_id)), [])
        if not rows:
            raise HTTPException(
                status_code=404, detail=f"district {district_id} not found"
            )

        thresholds = _predictor().bundle.horizons[horizon].risk_thresholds
        class_names = _predictor().class_names()
        out = []
        for row in rows:
            day = row["date"]
            if start and day < start:
                continue
            if end and day > end:
                continue
            mean_prob = row.get(f"mean_h{horizon}")
            max_prob = row.get(f"max_h{horizon}")
            if mean_prob is None:
                risk_idx = -1
            else:
                risk_idx = int(
                    classify(np.asarray([mean_prob], dtype=np.float64), thresholds)[0]
                )
            out.append(
                {
                    "date": day,
                    "mean_prob": mean_prob,
                    "max_prob": max_prob,
                    "risk_class": risk_idx,
                    "risk_class_name": class_names[risk_idx] if risk_idx >= 0 else "None",
                }
            )
        if not out:
            raise HTTPException(
                status_code=404, detail=f"district {district_id} not found"
            )
        return {"district_id": district_id, "horizon": horizon, "timeseries": out}

    return router_app


def router(root: Path | None = None) -> Any:
    if root is None:
        from prometheus.api.app import forecasts_root as root_fn  # type: ignore

        return router_factory(root_fn)
    return router_factory(lambda: root)


__all__ = ["router"]

