"""Isotonic calibration and reliability diagnostics for fire-risk probabilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Below this the isotonic fit has too few positives to be meaningful.
MIN_POSITIVES = 200


@dataclass
class Calibrator:
    """
    A frozen isotonic map from raw LightGBM score to probability.

    Stored as breakpoints rather than a pickled sklearn object so the bundle
    stays readable, portable across library versions, and cheap to apply — a
    single `np.interp` over the grid.
    """

    x: np.ndarray
    y: np.ndarray
    base_rate: float
    fit_year: int | None = None
    n_fit: int = 0

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        flat = np.asarray(scores, dtype=np.float64).ravel()
        out = np.interp(flat, self.x, self.y, left=self.y[0], right=self.y[-1])
        return np.clip(out, 0.0, 1.0).reshape(np.shape(scores)).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "base_rate": self.base_rate,
            "fit_year": self.fit_year,
            "n_fit": self.n_fit,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Calibrator:
        return cls(
            x=np.asarray(payload["x"], dtype=np.float64),
            y=np.asarray(payload["y"], dtype=np.float64),
            base_rate=float(payload["base_rate"]),
            fit_year=payload.get("fit_year"),
            n_fit=int(payload.get("n_fit", 0)),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_isotonic(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    fit_year: int | None = None,
    max_breakpoints: int = 512,
) -> Calibrator:
    """
    Fit isotonic regression on scores that carry the real base rate.

    This must be fit on full-grid predictions. The training table is downsampled
    to 1:20, so a calibrator fit on it would learn to map onto the *sampled*
    prevalence and stay wrong by more than an order of magnitude.
    """
    from sklearn.isotonic import IsotonicRegression

    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() < MIN_POSITIVES:
        raise ValueError(f"only {int(y.sum())} positives; need {MIN_POSITIVES}")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s, y)

    knots = np.asarray(iso.X_thresholds_, dtype=np.float64)
    values = np.asarray(iso.y_thresholds_, dtype=np.float64)
    if knots.size > max_breakpoints:
        # Thin the step function on a quantile grid: the curve is monotone and
        # smooth, so a few hundred knots reproduce it to well under 1e-4.
        idx = np.unique(
            np.linspace(0, knots.size - 1, max_breakpoints).round().astype(int)
        )
        knots, values = knots[idx], values[idx]

    return Calibrator(
        x=knots, y=values, base_rate=float(y.mean()), fit_year=fit_year, n_fit=int(y.size)
    )


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> dict[str, np.ndarray]:
    """
    Observed vs predicted frequency per bin.

    Equal-width bins are the textbook choice but useless for rare events — at a
    0.8 % base rate nearly every pixel lands in the first bin. Equal-count
    ("quantile") bins spread the mass across the range the model actually uses,
    so both are reported.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    if edges.size < 3:
        edges = np.linspace(0.0, max(p.max(), 1e-6), 3)

    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, edges.size - 2)
    counts = np.bincount(idx, minlength=edges.size - 1).astype(np.float64)
    pred_sum = np.bincount(idx, weights=p, minlength=edges.size - 1)
    obs_sum = np.bincount(idx, weights=y, minlength=edges.size - 1)

    keep = counts > 0
    return {
        "mean_predicted": pred_sum[keep] / counts[keep],
        "observed_frequency": obs_sum[keep] / counts[keep],
        "count": counts[keep],
        "edges": edges,
    }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 15, strategy: str = "quantile"
) -> float:
    """Count-weighted mean gap between predicted and observed frequency."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    gap = np.abs(curve["mean_predicted"] - curve["observed_frequency"])
    weights = curve["count"] / curve["count"].sum()
    return float((gap * weights).sum())


def calibration_report(
    y_true: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, *, n_bins: int = 15
) -> dict:
    """Before/after summary: ECE both ways, mean probability, Brier."""
    from prometheus.eval.metrics import brier, pr_auc

    y = np.asarray(y_true).ravel()
    out: dict = {
        "n": int(y.size),
        "base_rate": float(y.mean()),
        "mean_raw": float(np.mean(raw)),
        "mean_calibrated": float(np.mean(calibrated)),
    }
    for label, scores in (("raw", raw), ("calibrated", calibrated)):
        out[f"ece_quantile_{label}"] = expected_calibration_error(
            y, scores, n_bins=n_bins, strategy="quantile"
        )
        out[f"ece_uniform_{label}"] = expected_calibration_error(
            y, scores, n_bins=n_bins, strategy="uniform"
        )
        out[f"brier_{label}"] = brier(y, scores)
    # Isotonic is monotone, so ranking metrics are preserved up to ties; report
    # PR-AUC on both to prove calibration did not cost discrimination.
    out["pr_auc_raw"] = pr_auc(y, raw)
    out["pr_auc_calibrated"] = pr_auc(y, calibrated)
    return out


def quantile_thresholds(
    probabilities: np.ndarray, quantiles: list[float]
) -> list[float]:
    """
    Interior cut points for the risk classes.

    Quantiles are taken over the predicted distribution across the whole season
    and forest mask, so "Extreme" means the top 5 % of place-days rather than a
    fixed probability — the operational fire-danger convention.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    interior = [q for q in quantiles if 0.0 < q < 1.0]
    return [float(v) for v in np.quantile(p, interior)]


def classify(probabilities: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Map probabilities to risk-class indices 0..len(thresholds)."""
    p = np.asarray(probabilities, dtype=np.float64)
    out = np.digitize(p, np.asarray(thresholds, dtype=np.float64), right=False)
    return np.where(np.isfinite(p), out, -1).astype(np.int8)


__all__ = [
    "Calibrator",
    "calibration_report",
    "classify",
    "expected_calibration_error",
    "fit_isotonic",
    "quantile_thresholds",
    "reliability_curve",
]
