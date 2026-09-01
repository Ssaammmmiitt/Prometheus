"""What-if scoring: one forest cell, optional weather overrides, calibrated chance."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from rasterio.transform import rowcol, xy

from prometheus.api.runtime import predictor as get_predictor
from prometheus.features import forest
from prometheus.features import table as ftable
from prometheus.features.weather import dewpoint_from_rh, vapour_pressure_deficit
from prometheus.grid import transform as grid_transform
from prometheus.models import lgbm
from prometheus.models.calibrate import classify
from prometheus.models.predict import _as_date

router_app = APIRouter()

SLIDER_FEATURES = (
    "t2m_max",
    "t2m",
    "t2m_min",
    "rh",
    "precip",
    "precip_7d",
    "precip_30d",
    "consecutive_dry_days",
    "days_since_rain",
    "wind_speed",
    "wind_max_7d",
    "ndvi",
    "ndvi_anomaly",
    "lst_day",
    "lst_night",
)

SLIDER_META = {
    "t2m_max": {"group": "temperature", "unit": "°C", "step": 0.5},
    "t2m": {"group": "temperature", "unit": "°C", "step": 0.5},
    "t2m_min": {"group": "temperature", "unit": "°C", "step": 0.5},
    "rh": {"group": "moisture", "unit": "%", "step": 1.0},
    "precip": {"group": "rain", "unit": "mm", "step": 0.5},
    "precip_7d": {"group": "rain", "unit": "mm", "step": 1.0},
    "precip_30d": {"group": "rain", "unit": "mm", "step": 1.0},
    "consecutive_dry_days": {"group": "rain", "unit": "days", "step": 1.0},
    "days_since_rain": {"group": "rain", "unit": "days", "step": 1.0},
    "wind_speed": {"group": "wind", "unit": "m/s", "step": 0.1},
    "wind_max_7d": {"group": "wind", "unit": "m/s", "step": 0.1},
    "ndvi": {"group": "plants", "unit": "", "step": 0.01},
    "ndvi_anomaly": {"group": "plants", "unit": "", "step": 0.01},
    "lst_day": {"group": "ground", "unit": "°C", "step": 0.5},
    "lst_night": {"group": "ground", "unit": "°C", "step": 0.5},
}

PLACE_FACTS = (
    "elevation",
    "slope",
    "tree_frac",
    "shrub_frac",
    "grass_frac",
    "crop_frac",
    "dist_road",
    "dist_settlement",
    "fire_clim",
    "days_since_fire",
    "fires_1yr",
    "fires_3yr",
    "fires_5yr",
)

LOCKED = set(PLACE_FACTS) | {
    "aspect_sin",
    "aspect_cos",
    "twi",
    "built_frac",
    "doy_sin",
    "doy_cos",
    "surface_pressure",
    "soil_water_l1",
}

FALLBACK_BOUNDS = {
    "t2m_max": (-8.0, 38.0),
    "t2m": (-13.0, 32.0),
    "t2m_min": (-22.0, 24.0),
    "rh": (10.0, 100.0),
    "precip": (0.0, 40.0),
    "precip_7d": (0.0, 120.0),
    "precip_30d": (0.0, 400.0),
    "consecutive_dry_days": (0.0, 150.0),
    "days_since_rain": (0.0, 150.0),
    "wind_speed": (0.0, 12.0),
    "wind_max_7d": (0.0, 15.0),
    "ndvi": (-0.1, 0.9),
    "ndvi_anomaly": (-0.4, 0.4),
    "lst_day": (-10.0, 50.0),
    "lst_night": (-20.0, 30.0),
}


class WhatIfRequest(BaseModel):
    lat: float
    lon: float
    date: str
    horizon: int = Field(1, ge=1, le=7)
    top: int = Field(8, ge=1, le=12)
    overrides: dict[str, float] = Field(default_factory=dict)


def _twin_partner(feature: str) -> str | None:
    for a, b, _ in lgbm.COLLINEAR_TWINS:
        if feature == a:
            return b
        if feature == b:
            return a
    return None


def _cell_pos(pred, r: int, c: int) -> int:
    mask = (pred._rows == r) & (pred._cols == c)
    idx = np.flatnonzero(mask)
    if idx.size != 1:
        raise HTTPException(status_code=400, detail="cell not on forest mask")
    return int(idx[0])


def _locate(pred, lat: float, lon: float) -> tuple[int, int, int]:
    r, c = rowcol(grid_transform(), lon, lat)
    if r < 0 or c < 0 or r >= pred.shape[0] or c >= pred.shape[1]:
        raise HTTPException(status_code=400, detail="lat/lon outside grid")
    if not bool(forest.forest_mask()[r, c]):
        raise HTTPException(
            status_code=400,
            detail="lat/lon outside forest mask",
        )
    return int(r), int(c), _cell_pos(pred, int(r), int(c))


def training_bounds() -> dict[str, tuple[float, float]]:
    path = ftable.norm_stats_path()
    if not path.is_file():
        return dict(FALLBACK_BOUNDS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for feat in payload.get("features", []):
        lows, highs = [], []
        for fold in payload.get("folds", {}).values():
            stats = fold.get(feat)
            if not stats:
                continue
            lows.append(float(stats["p1"]))
            highs.append(float(stats["p99"]))
        if lows:
            lo, hi = min(lows), max(highs)
            if hi <= lo:
                hi = lo + 1e-3
            out[feat] = (lo, hi)
    for feat, pair in FALLBACK_BOUNDS.items():
        out.setdefault(feat, pair)
    return out


def _clip(name: str, value: float, bounds: dict[str, tuple[float, float]]) -> tuple[float, bool]:
    lo, hi = bounds.get(name, (float(value), float(value)))
    clipped = float(np.clip(value, lo, hi))
    return clipped, clipped != float(value) and abs(clipped - value) > 1e-6


def apply_overrides(
    values: dict[str, float],
    overrides: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], list[str], list[str]]:
    """Copy a cell, apply slider overrides, then restore physical couples."""
    out = dict(values)
    clamped: list[str] = []
    ignored: list[str] = []

    for name, raw in overrides.items():
        if name in LOCKED or name not in out:
            ignored.append(name)
            continue
        if name not in SLIDER_FEATURES:
            ignored.append(name)
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            ignored.append(name)
            continue
        clipped, hit = _clip(name, number, bounds)
        out[name] = clipped
        if hit:
            clamped.append(name)

    if out.get("t2m_min", 0) > out.get("t2m", 0):
        out["t2m"] = out["t2m_min"]
    if out.get("t2m", 0) > out.get("t2m_max", 0):
        out["t2m_max"] = out["t2m"]
    if out.get("t2m_min", 0) > out.get("t2m", 0):
        out["t2m_min"] = out["t2m"]

    if "t2m_max_7d" in out:
        out["t2m_max_7d"] = max(float(out["t2m_max_7d"]), float(out.get("t2m_max", 0)))
    if "rh_min_7d" in out:
        out["rh_min_7d"] = min(float(out["rh_min_7d"]), float(out.get("rh", 100)))

    if "precip_7d" in out and "precip_30d" in out:
        if out["precip_7d"] > out["precip_30d"]:
            out["precip_30d"] = out["precip_7d"]
    if "precip" in out:
        out["precip"] = max(float(out["precip"]), 0.0)

    t2m = float(out.get("t2m", 0.0))
    rh = float(np.clip(out.get("rh", 50.0), 1.0, 100.0))
    out["rh"] = rh
    out["vpd"] = float(vapour_pressure_deficit(np.array(t2m), np.array(rh)))
    out["d2m"] = float(dewpoint_from_rh(np.array(t2m), np.array(rh)))

    if "lst_day" in out and "lst_night" in out:
        if out["lst_night"] > out["lst_day"]:
            out["lst_night"] = out["lst_day"]
        out["lst_diff"] = float(out["lst_day"] - out["lst_night"])

    speed = float(out.get("wind_speed", 0.0))
    if "wind_max_7d" in out:
        out["wind_max_7d"] = max(float(out["wind_max_7d"]), speed)
    old = float(values.get("wind_speed", 0.0)) or 1e-6
    scale = speed / old
    if "u10" in out:
        out["u10"] = float(out["u10"]) * scale
    if "v10" in out:
        out["v10"] = float(out["v10"]) * scale

    if "ndvi" in out and "evi" in out:
        delta = float(out["ndvi"]) - float(values.get("ndvi", out["ndvi"]))
        out["evi"] = float(np.clip(float(out["evi"]) + delta, -0.2, 1.0))

    return out, clamped, ignored


def _vector(feat_order: list[str], values: dict[str, float]) -> np.ndarray:
    return np.array([float(values[name]) for name in feat_order], dtype=np.float32)[None, :]


def _score(booster, artifacts, values: dict[str, float], top: int):
    wanted = artifacts.features
    x = _vector(wanted, {n: values[n] for n in wanted})
    raw = float(booster.predict(x, **lgbm._predict_kwargs(booster))[0])
    probability = float(artifacts.calibrator(np.array([raw], dtype=np.float32))[0])
    names = get_predictor().class_names()
    idx = int(classify(np.array([probability]), artifacts.risk_thresholds)[0])
    contrib = booster.predict(
        x, pred_contrib=True, **lgbm._predict_kwargs(booster)
    )[0]
    shap_vals = contrib[:-1]
    order = np.argsort(-np.abs(shap_vals))[:top]
    top_feats = []
    for i in order:
        name = wanted[i]
        top_feats.append(
            {
                "feature": name,
                "value": float(x[0, i]),
                "shap_value": float(shap_vals[i]),
                "abs_shap": float(abs(shap_vals[i])),
                "collinear_twin": _twin_partner(name),
            }
        )
    return {
        "raw": raw,
        "probability": probability,
        "risk_class": idx,
        "risk_class_name": names[idx] if 0 <= idx < len(names) else "None",
        "expected_value": float(contrib[-1]),
        "top": top_feats,
    }


def _pack_features(feat_order: list[str], values: dict[str, float]) -> dict[str, float]:
    return {name: float(values[name]) for name in feat_order}


def router() -> APIRouter:
    @router_app.get("/whatif/schema")
    def schema() -> dict[str, Any]:
        bounds = training_bounds()
        sliders = []
        for name in SLIDER_FEATURES:
            lo, hi = bounds.get(name, FALLBACK_BOUNDS.get(name, (0.0, 1.0)))
            meta = SLIDER_META[name]
            sliders.append(
                {
                    "feature": name,
                    "group": meta["group"],
                    "unit": meta["unit"],
                    "step": meta["step"],
                    "lo": float(lo),
                    "hi": float(hi),
                }
            )
        return {
            "sliders": sliders,
            "place_facts": list(PLACE_FACTS),
            "derived": ["vpd", "d2m", "lst_diff"],
            "note": (
                "Pick a forest cell first. Sliders stay inside training "
                "percentiles. VPD is computed from temperature and humidity."
            ),
        }

    @router_app.post("/whatif")
    def whatif(body: WhatIfRequest) -> dict[str, Any]:
        if body.horizon not in (1, 7):
            raise HTTPException(status_code=400, detail="horizon must be 1 or 7")
        pred = get_predictor()
        day = _as_date(body.date)
        season = pred._year_features(day.year)
        if day not in season["day_index"]:
            raise HTTPException(status_code=400, detail="date outside model season")

        r, c, cell_pos = _locate(pred, body.lat, body.lon)
        t = season["day_index"][day]
        n_cells = season["n_cells"]
        feat_order = list(season["features"])
        row = season["matrix"][t * n_cells : (t + 1) * n_cells][cell_pos]
        baseline_vals = {name: float(row[i]) for i, name in enumerate(feat_order)}

        bounds = training_bounds()
        scenario_vals, clamped, ignored = apply_overrides(
            baseline_vals, body.overrides, bounds
        )

        artifacts = pred.bundle.horizons[body.horizon]
        booster = pred._boosters[body.horizon]
        baseline = _score(booster, artifacts, baseline_vals, body.top)
        scenario = _score(booster, artifacts, scenario_vals, body.top)
        lon_c, lat_c = xy(grid_transform(), r, c, offset="center")

        return {
            "date": day.isoformat(),
            "horizon": body.horizon,
            "grid_cell": {
                "row": r,
                "col": c,
                "lat": float(lat_c),
                "lon": float(lon_c),
                "forest_cell_index": cell_pos,
            },
            "baseline": baseline,
            "scenario": scenario,
            "features": _pack_features(feat_order, scenario_vals),
            "place": {k: baseline_vals[k] for k in PLACE_FACTS if k in baseline_vals},
            "clamped": clamped,
            "ignored": ignored,
            "base_rate": float(artifacts.calibrator.base_rate),
            "note": (
                "This is a calibrated chance of a satellite fire detection, "
                "not a yes/no that a village will burn. Extreme is still often "
                "only a few percent."
            ),
        }

    return router_app


__all__ = ["router"]
