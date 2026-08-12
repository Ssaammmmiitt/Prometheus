"""Score saved h1 forecasts against next-day FIRMS detections."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from prometheus.eval import metrics
from prometheus.features import forest
from prometheus.features import table as ftable
from prometheus.infer import io_cog
from prometheus.models.predict import _as_date


def verification_path(root: Path | None = None) -> Path:
    return (root or io_cog.forecasts_dir()) / "verification.csv"


def next_day_fire(day: date) -> np.ndarray:
    """Binary fire field on day+1 (the target of a day-D h1 forecast)."""
    fire_ds = ftable._fire_cube()
    observe = day + timedelta(days=1)
    try:
        return fire_ds["fire"].sel(time=str(observe)).values.astype(np.uint8)
    except KeyError as exc:
        raise KeyError(f"no fire labels for observation day {observe}") from exc


def score_forecast_day(
    forecast_day: date | str,
    *,
    root: Path | None = None,
) -> dict:
    """PR-AUC / Brier / top-10% of the saved h1 COG vs next-day detections."""
    day = _as_date(forecast_day)
    path = io_cog.risk_path(day, 1, root)
    if not path.is_file():
        raise FileNotFoundError(f"missing forecast COG {path}")

    risk = io_cog.read_risk(path)
    observed = next_day_fire(day)
    mask = forest.forest_mask() & np.isfinite(risk)

    y = observed[mask].ravel().astype(np.float64)
    p = risk[mask].ravel().astype(np.float64)
    if y.size == 0 or y.sum() == 0:
        # Quiet day: still record the row so the series is continuous.
        return {
            "forecast_date": day.isoformat(),
            "observe_date": (day + timedelta(days=1)).isoformat(),
            "n": int(y.size),
            "n_pos": int(y.sum()),
            "base_rate": float(y.mean()) if y.size else float("nan"),
            "mean_forecast": float(p.mean()) if p.size else float("nan"),
            "pr_auc": float("nan"),
            "brier": metrics.brier(y, p) if y.size else float("nan"),
            "top10_capture": float("nan"),
            "valid": False,
        }

    return {
        "forecast_date": day.isoformat(),
        "observe_date": (day + timedelta(days=1)).isoformat(),
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()),
        "mean_forecast": float(p.mean()),
        "pr_auc": metrics.pr_auc(y, p),
        "brier": metrics.brier(y, p),
        "top10_capture": metrics.top_k_capture(y, p, 0.10),
        "valid": True,
    }


def verify_range(
    start: date | str,
    end: date | str,
    *,
    root: Path | None = None,
    append: bool = True,
) -> pd.DataFrame:
    """
    Score every day in [start, end] that has a saved h1 forecast and a next-day label.

    Idempotent: re-running rewrites rows for those dates rather than duplicating them.
    """
    root = root or io_cog.forecasts_dir()
    start_d, end_d = _as_date(start), _as_date(end)
    rows = []
    d = start_d
    while d <= end_d:
        cog = io_cog.risk_path(d, 1, root)
        if cog.is_file():
            try:
                rows.append(score_forecast_day(d, root=root))
            except KeyError:
                pass  # last day of a season has no next-day fire layer
        d += timedelta(days=1)

    frame = pd.DataFrame(rows)
    out = verification_path(root)
    if append and out.is_file() and not frame.empty:
        prior = pd.read_csv(out)
        prior = prior[~prior["forecast_date"].isin(set(frame["forecast_date"]))]
        frame = pd.concat([prior, frame], ignore_index=True)
        frame = frame.sort_values("forecast_date")
    if not frame.empty:
        frame.to_csv(out, index=False)
    return frame


__all__ = [
    "next_day_fire",
    "score_forecast_day",
    "verification_path",
    "verify_range",
]
