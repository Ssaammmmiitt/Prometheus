"""One-cell explanation: calibrated chance, comparisons, snapshot, grouped SHAP."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from rasterio.transform import rowcol, xy

from prometheus.api.runtime import predictor as get_predictor
from prometheus.features import forest
from prometheus.features.derived import NO_FIRE_SENTINEL
from prometheus.grid import transform as grid_transform
from prometheus.infer import io_cog
from prometheus.infer.districts import district_id_raster
from prometheus.models import lgbm
from prometheus.models.calibrate import classify
from prometheus.models.predict import _as_date

router_app = APIRouter()

# Group every SHAP coordinate into a handful of stories the drawer can chart.
# Collinear twins (t2m/t2m_max, doy_sin/doy_cos, fires_1yr/3yr/5yr) collapse
# so the panel does not repeat "days since last fire" three times.
DRIVER_GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "heat": ("Heat", ("t2m_max", "t2m", "t2m_min", "t2m_max_7d", "lst_day", "lst_night", "lst_diff")),
    "moisture": ("Humidity and dry air", ("rh", "rh_min_7d", "vpd", "d2m", "soil_water_l1")),
    "rain": ("Rain and dry spell", ("precip", "precip_7d", "precip_30d", "consecutive_dry_days", "days_since_rain")),
    "wind": ("Wind", ("wind_speed", "wind_max_7d", "u10", "v10")),
    "fire": ("Fire history", ("fire_clim", "days_since_fire", "fires_1yr", "fires_3yr", "fires_5yr")),
    "plants": ("Plant greenness", ("ndvi", "evi", "ndvi_anomaly")),
    "season": ("Time of year", ("doy_sin", "doy_cos")),
    "terrain": ("Terrain", ("elevation", "surface_pressure", "slope", "twi", "aspect_sin", "aspect_cos")),
    "cover": ("Land cover", ("tree_frac", "shrub_frac", "grass_frac", "crop_frac", "built_frac")),
    "access": ("Roads and settlements", ("dist_road", "dist_settlement")),
}

_FEATURE_TO_GROUP = {
    feat: gid for gid, (_, feats) in DRIVER_GROUPS.items() for feat in feats
}

FEATURE_LABELS = {
    "t2m_max": "Hottest today",
    "t2m": "Air temperature",
    "t2m_min": "Coldest today",
    "rh": "Humidity",
    "vpd": "Air dryness (VPD)",
    "precip": "Rain today",
    "precip_7d": "Rain, last 7 days",
    "precip_30d": "Rain, last 30 days",
    "consecutive_dry_days": "Days without rain",
    "days_since_rain": "Days since measurable rain",
    "wind_speed": "Wind",
    "ndvi": "Greenness (NDVI)",
    "ndvi_anomaly": "Greenness vs a normal year",
    "days_since_fire": "Days since last fire",
    "fire_clim": "Usual fire rate this date",
    "fires_1yr": "Fires in the last year",
    "fires_3yr": "Fires in the last 3 years",
    "fires_5yr": "Fires in the last 5 years",
    "elevation": "Elevation",
    "tree_frac": "Tree cover",
    "slope": "Steepness",
}

# Snapshot order: weather people can check, then fuel, then history.
SNAPSHOT_KEYS = (
    "t2m_max",
    "rh",
    "vpd",
    "precip_7d",
    "consecutive_dry_days",
    "wind_speed",
    "ndvi",
    "days_since_fire",
    "fire_clim",
    "fires_5yr",
    "elevation",
    "tree_frac",
)


def _cell_pos(pred, r: int, c: int) -> int:
    mask = (pred._rows == r) & (pred._cols == c)
    idx = np.flatnonzero(mask)
    if idx.size != 1:
        raise HTTPException(status_code=400, detail="cell not on forest mask")
    return int(idx[0])


def _twin_partner(feature: str) -> str | None:
    for a, b, _ in lgbm.COLLINEAR_TWINS:
        if feature == a:
            return b
        if feature == b:
            return a
    return None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def format_snapshot_item(name: str, raw: float) -> dict[str, Any] | None:
    """Turn one model feature into a labelled, unit-bearing figure."""
    value = _finite(raw)
    if value is None:
        return None
    label = FEATURE_LABELS.get(name, name.replace("_", " "))

    if name == "days_since_fire":
        if value >= NO_FIRE_SENTINEL * 0.9:
            return {
                "key": name,
                "label": "Last recorded fire",
                "display": "None this season",
                "unit": "",
                "value": None,
            }
        return {
            "key": name,
            "label": label,
            "display": f"{int(round(value)):,}",
            "unit": "days",
            "value": float(value),
        }

    if name == "rh":
        return {
            "key": name,
            "label": label,
            "display": f"{value:.0f}",
            "unit": "%",
            "value": float(value),
        }

    if name in ("tree_frac", "shrub_frac", "grass_frac", "crop_frac", "built_frac"):
        pct = value * 100.0 if value <= 1.5 else value
        return {
            "key": name,
            "label": label,
            "display": f"{pct:.0f}",
            "unit": "%",
            "value": float(value),
        }

    if name == "fire_clim":
        return {
            "key": name,
            "label": label,
            "display": f"{value * 100.0:.2f}",
            "unit": "% of cells historically",
            "value": float(value),
        }

    if name == "vpd":
        return {
            "key": name,
            "label": label,
            "display": f"{value:.2f}",
            "unit": "kPa",
            "value": float(value),
        }

    if name.startswith("precip"):
        return {
            "key": name,
            "label": label,
            "display": f"{value:.1f}",
            "unit": "mm",
            "value": float(value),
        }

    if name.startswith("t2m") or name.startswith("lst"):
        return {
            "key": name,
            "label": label,
            "display": f"{value:.1f}",
            "unit": "°C",
            "value": float(value),
        }

    if name.startswith("wind"):
        return {
            "key": name,
            "label": label,
            "display": f"{value:.1f}",
            "unit": "m/s",
            "value": float(value),
        }

    if name == "elevation":
        return {
            "key": name,
            "label": label,
            "display": f"{int(round(value)):,}",
            "unit": "m",
            "value": float(value),
        }

    if name.startswith("ndvi") or name == "evi":
        sign = "+" if value > 0 and "anomaly" in name else ""
        return {
            "key": name,
            "label": label,
            "display": f"{sign}{value:.2f}",
            "unit": "",
            "value": float(value),
        }

    if name.startswith("fires_") or "days" in name:
        return {
            "key": name,
            "label": label,
            "display": f"{int(round(value)):,}",
            "unit": "count" if name.startswith("fires_") else "days",
            "value": float(value),
        }

    return {
        "key": name,
        "label": label,
        "display": f"{value:.2f}",
        "unit": "",
        "value": float(value),
    }


def snapshot_from_values(values: dict[str, float], limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in SNAPSHOT_KEYS:
        if name not in values:
            continue
        item = format_snapshot_item(name, values[name])
        if item is not None:
            rows.append(item)
        if len(rows) >= limit:
            break
    return rows


def grouped_drivers(
    feature_names: list[str],
    shap_vals: np.ndarray,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Net SHAP per theme. Shares are of the themes shown, so they sum to 1."""
    buckets: dict[str, dict[str, Any]] = {}
    for name, shap in zip(feature_names, shap_vals, strict=False):
        gid = _FEATURE_TO_GROUP.get(name)
        if gid:
            label = DRIVER_GROUPS[gid][0]
            key = gid
        else:
            key = name
            label = FEATURE_LABELS.get(name, name.replace("_", " "))
        slot = buckets.get(key)
        if slot is None:
            buckets[key] = {"key": key, "label": label, "shap": float(shap)}
        else:
            slot["shap"] += float(shap)

    rows = list(buckets.values())
    for row in rows:
        row["abs_shap"] = abs(row["shap"])
        row["direction"] = "up" if row["shap"] >= 0 else "down"
    rows.sort(key=lambda row: row["abs_shap"], reverse=True)
    shown = rows[:limit]
    total_shown = sum(row["abs_shap"] for row in shown) or 1.0
    for row in shown:
        row["share"] = row["abs_shap"] / total_shown
    return shown


def country_compare(day, horizon: int, row: int, col: int, probability: float) -> dict[str, Any]:
    """This cell vs every forest cell on today's saved risk map."""
    path = io_cog.risk_path(day, horizon)
    empty = {
        "mean": None,
        "median": None,
        "percentile": None,
        "n_forest": None,
        "this_cell": float(probability),
    }
    if not path.is_file():
        return empty
    risk = io_cog.read_risk(path)
    mask = forest.forest_mask()
    vals = risk[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return empty
    cell = risk[row, col]
    cell_p = float(cell) if np.isfinite(cell) else float(probability)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "percentile": float(100.0 * np.mean(vals <= cell_p)),
        "n_forest": int(vals.size),
        "this_cell": cell_p,
    }


def district_context(day, horizon: int, row: int, col: int) -> dict[str, Any] | None:
    codes = district_id_raster()
    did = int(codes[row, col])
    if did <= 0:
        return None
    out: dict[str, Any] = {"district_id": did, "name": None, "mean": None, "max": None}
    path = io_cog.districts_path(day)
    if not path.is_file():
        return out
    payload = json.loads(path.read_text(encoding="utf-8"))
    for feat in payload.get("features", []):
        props = feat.get("properties") or {}
        try:
            if int(props.get("district_id", -1)) != did:
                continue
        except (TypeError, ValueError):
            continue
        out["name"] = props.get("name")
        out["mean"] = _finite(props.get(f"mean_h{horizon}"))
        out["max"] = _finite(props.get(f"max_h{horizon}"))
        out["n_forest"] = int(props["n_forest_cells"]) if props.get("n_forest_cells") is not None else None
        out["risk_class_name"] = props.get("risk_class_name")
        return out
    return out


def compare_rows(
    probability: float,
    base_rate: float,
    country: dict[str, Any],
    district: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = [
        {
            "id": "here",
            "label": "This cell",
            "probability": float(probability),
        }
    ]
    if district and district.get("mean") is not None:
        name = district.get("name") or "District"
        rows.append(
            {
                "id": "district",
                "label": f"{name} avg",
                "probability": float(district["mean"]),
            }
        )
    if country.get("mean") is not None:
        rows.append(
            {
                "id": "country",
                "label": "Nepal forest today",
                "probability": float(country["mean"]),
            }
        )
    rows.append(
        {
            "id": "typical",
            "label": "Typical forest day",
            "probability": float(base_rate),
        }
    )
    return rows


def headline(
    probability: float,
    base_rate: float,
    percentile: float | None,
    horizon: int,
) -> str:
    window = "tomorrow" if horizon == 1 else "over the next 7 days"
    p = f"{probability * 100.0:.2f}%"
    parts = [f"{p} chance of a satellite fire detection {window}."]
    if base_rate > 1e-9:
        ratio = probability / base_rate
        typical = f"{base_rate * 100.0:.2f}%"
        if ratio >= 1.2:
            parts.append(f"About {ratio:.1f}× a typical forest cell ({typical}).")
        elif ratio <= 0.8:
            parts.append(f"About {1.0 / ratio:.1f}× below a typical forest cell ({typical}).")
        else:
            parts.append(f"Close to a typical forest cell ({typical}).")
    if percentile is not None:
        if percentile >= 70:
            parts.append(f"Higher than {percentile:.0f}% of forest cells in Nepal today.")
        elif percentile <= 30:
            parts.append(f"Lower than most forest cells today ({percentile:.0f}th percentile).")
        else:
            parts.append(f"Around the middle of Nepal's forest today ({percentile:.0f}th percentile).")
    return " ".join(parts)


def router() -> APIRouter:
    @router_app.get("/explain")
    def explain(
        lat: float = Query(...),
        lon: float = Query(...),
        date: str = Query(..., description="YYYY-MM-DD"),
        horizon: int = Query(1, ge=1, le=7),
        top: int = Query(6, ge=1, le=12),
    ) -> dict[str, Any]:
        pred = get_predictor()
        day = _as_date(date)
        season = pred._year_features(day.year)

        if horizon not in pred.bundle.horizons:
            raise HTTPException(status_code=400, detail="horizon must be 1 or 7")
        if day not in season["day_index"]:
            raise HTTPException(status_code=400, detail="date outside model season")
        t = season["day_index"][day]

        r, c = rowcol(grid_transform(), lon, lat)
        if r < 0 or c < 0 or r >= pred.shape[0] or c >= pred.shape[1]:
            raise HTTPException(status_code=400, detail="lat/lon outside grid")

        mask = forest.forest_mask()
        if not bool(mask[r, c]):
            raise HTTPException(status_code=400, detail="lat/lon outside forest mask")

        cell_pos = _cell_pos(pred, r, c)
        artifacts = pred.bundle.horizons[horizon]
        booster = pred._boosters[horizon]

        n_cells = season["n_cells"]
        block = season["matrix"][t * n_cells : (t + 1) * n_cells][cell_pos]
        feat_order = list(season["features"])
        values = {name: float(block[i]) for i, name in enumerate(feat_order)}
        wanted = artifacts.features
        x = np.array([values[name] for name in wanted], dtype=np.float32)[None, :]

        raw = float(booster.predict(x, **lgbm._predict_kwargs(booster))[0])
        probability = float(artifacts.calibrator(np.array([raw], dtype=np.float32))[0])
        names = pred.class_names()
        class_idx = int(classify(np.array([probability]), artifacts.risk_thresholds)[0])
        class_name = names[class_idx] if 0 <= class_idx < len(names) else "None"
        base_rate = float(artifacts.calibrator.base_rate)

        contrib = booster.predict(
            x,
            pred_contrib=True,
            **lgbm._predict_kwargs(booster),  # type: ignore[attr-defined]
        )[0]
        expected = float(contrib[-1])
        shap_vals = contrib[:-1]

        order = np.argsort(-np.abs(shap_vals))[:top]
        top_feats = []
        for idx in order:
            name = wanted[idx]
            top_feats.append(
                {
                    "feature": name,
                    "value": float(x[0, idx]),
                    "shap_value": float(shap_vals[idx]),
                    "abs_shap": float(abs(shap_vals[idx])),
                    "collinear_twin": _twin_partner(name),
                }
            )

        country = country_compare(day, horizon, int(r), int(c), probability)
        district = district_context(day, horizon, int(r), int(c))
        lon_c, lat_c = xy(grid_transform(), r, c, offset="center")

        return {
            "date": day.isoformat(),
            "horizon": horizon,
            "grid_cell": {
                "row": int(r),
                "col": int(c),
                "lat": float(lat_c),
                "lon": float(lon_c),
                "forest_cell_index": cell_pos,
            },
            "probability": probability,
            "raw": raw,
            "risk_class": class_idx,
            "risk_class_name": class_name,
            "base_rate": base_rate,
            "vs_country": country,
            "district": district,
            "compare": compare_rows(probability, base_rate, country, district),
            "snapshot": snapshot_from_values(values),
            "drivers": grouped_drivers(list(wanted), shap_vals, limit=6),
            "headline": headline(probability, base_rate, country.get("percentile"), horizon),
            "expected_value": expected,
            "top": top_feats,
            "note": (
                "This is a calibrated chance of a satellite fire detection in "
                "this 1 km cell, not a yes/no that a village will burn. Extreme "
                "is still often only a few percent."
            ),
        }

    return router_app


__all__ = [
    "compare_rows",
    "country_compare",
    "format_snapshot_item",
    "grouped_drivers",
    "headline",
    "router",
    "snapshot_from_values",
]
