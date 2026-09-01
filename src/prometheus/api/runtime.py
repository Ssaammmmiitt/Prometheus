"""Shared FastAPI runtime: one RiskPredictor so Explain and What-if share RAM."""

from __future__ import annotations

from functools import lru_cache

from prometheus.models.predict import RiskPredictor


@lru_cache(maxsize=1)
def predictor() -> RiskPredictor:
    return RiskPredictor("latest")
