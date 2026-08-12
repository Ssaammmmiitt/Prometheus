from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from rasterio.transform import rowcol

from prometheus.features import forest
from prometheus.grid import transform as grid_transform
from prometheus.models import lgbm
from prometheus.models.predict import RiskPredictor, _as_date

router_app = APIRouter()


@lru_cache(maxsize=1)
def _predictor() -> RiskPredictor:
    return RiskPredictor("latest")


def _cell_pos(pred: RiskPredictor, r: int, c: int) -> int:
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


def router() -> APIRouter:
    @router_app.get("/explain")
    def explain(
        lat: float = Query(...),
        lon: float = Query(...),
        date: str = Query(..., description="YYYY-MM-DD"),
        horizon: int = Query(1, ge=1, le=7),
        top: int = Query(6, ge=1, le=12),
    ) -> dict[str, Any]:
        pred = _predictor()
        day = _as_date(date)
        season = pred._year_features(day.year)

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

        # Build a (1, n_features) feature vector for exactly this day+cell.
        n_cells = season["n_cells"]
        block = season["matrix"][t * n_cells : (t + 1) * n_cells][cell_pos]
        feat_order = season["features"]
        index_map = {nm: i for i, nm in enumerate(feat_order)}
        wanted = artifacts.features
        cols = [index_map[f] for f in wanted]
        x = np.asarray(block[cols], dtype=np.float32)[None, :]

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
            twin = _twin_partner(name)
            top_feats.append(
                {
                    "feature": name,
                    "value": float(x[0, idx]),
                    "shap_value": float(shap_vals[idx]),
                    "abs_shap": float(abs(shap_vals[idx])),
                    "collinear_twin": twin,
                }
            )

        return {
            "date": day.isoformat(),
            "horizon": horizon,
            "grid_cell": {"row": int(r), "col": int(c), "forest_cell_index": cell_pos},
            "expected_value": expected,
            "top": top_feats,
            "note": (
                "For collinear twins (e.g., t2m_max/t2m), SHAP credit is split; "
                "interpret at the pair level."
            ),
        }

    return router_app


__all__ = ["router"]

