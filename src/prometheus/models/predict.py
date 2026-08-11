"""Operational inference: a calibrated risk map for one date."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np

from prometheus import grid
from prometheus.features import forest
from prometheus.features import table as ftable
from prometheus.models import lgbm
from prometheus.models.bundle import ModelBundle
from prometheus.models.calibrate import classify


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


class RiskPredictor:
    """
    Calibrated risk maps from a frozen bundle.

    Season features are built once per year and cached, because the rolling
    windows, dry-day counters, and fire-history state for any single day depend on
    the whole season up to that point — there is no cheaper honest way to get one
    day in isolation. After the first call for a year, a date costs one array
    slice plus one booster pass.
    """

    def __init__(self, bundle: ModelBundle | str | Path = "latest"):
        self.bundle = (
            bundle if isinstance(bundle, ModelBundle) else ModelBundle.load(bundle)
        )
        self._boosters = {h: self.bundle.booster(h) for h in self.bundle.horizons}
        self._cache: dict[int, dict] = {}
        self._cube = None
        self._fire_ds = None
        self._anomaly = None
        mask = forest.forest_mask()
        self._rows, self._cols = np.where(mask)
        self.shape = grid.shape()

    @property
    def horizons(self) -> list[int]:
        return sorted(self._boosters)

    def _year_features(self, year: int) -> dict:
        cached = self._cache.get(year)
        if cached is not None:
            return cached
        if self._cube is None:
            self._cube = ftable.open_cube()
            self._fire_ds = ftable._fire_cube()
            self._anomaly = ftable.build_anomaly(self._cube, self.bundle.train_years)
        features = self.bundle.horizons[self.horizons[0]].features
        bundle = ftable.year_grid_features(
            year,
            cube=self._cube,
            fire_ds=self._fire_ds,
            anomaly=self._anomaly,
            features=features,
        )
        bundle["day_index"] = {d: i for i, d in enumerate(bundle["dates"])}
        self._cache = {year: bundle}  # one season at a time keeps memory bounded
        return bundle

    def warm(self, year: int) -> None:
        """Build and cache a season ahead of time so later calls stay sub-second."""
        self._year_features(year)

    def predict(
        self, when: date | datetime | str, horizon: int = 1, *, calibrated: bool = True
    ) -> np.ndarray:
        """Calibrated fire probability on the canonical grid; NaN off the mask."""
        day = _as_date(when)
        season = self._year_features(day.year)
        if day not in season["day_index"]:
            raise KeyError(
                f"{day} is outside the modelled season "
                f"({season['dates'][0]}..{season['dates'][-1]})"
            )
        t = season["day_index"][day]

        artifacts = self.bundle.horizons[horizon]
        booster = self._boosters[horizon]
        index = {name: i for i, name in enumerate(season["features"])}
        cols = np.array([index[f] for f in artifacts.features], dtype=np.intp)

        n_cells = season["n_cells"]
        block = season["matrix"][t * n_cells : (t + 1) * n_cells][:, cols]
        scores = booster.predict(block, **lgbm._predict_kwargs(booster))
        if calibrated:
            scores = artifacts.calibrator(scores)

        out = np.full(self.shape, np.nan, dtype=np.float32)
        out[self._rows, self._cols] = scores
        return out

    def risk_classes(self, when, horizon: int = 1) -> np.ndarray:
        """Risk-class indices (0..4); -1 off the forest mask."""
        probabilities = self.predict(when, horizon=horizon)
        return classify(probabilities, self.bundle.horizons[horizon].risk_thresholds)

    def class_names(self) -> list[str]:
        return list(self.bundle.risk_class_names)


def predict(
    when: date | datetime | str,
    horizon: int = 1,
    *,
    bundle: ModelBundle | str | Path = "latest",
) -> np.ndarray:
    """One-shot convenience wrapper; prefer `RiskPredictor` for repeated calls."""
    return RiskPredictor(bundle).predict(when, horizon=horizon)


__all__ = ["RiskPredictor", "predict"]
