"""Evaluation metrics for wildfire risk maps."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def _to_1d(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).ravel().astype(np.float64)
    s = np.asarray(y_score).ravel().astype(np.float64)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch y={y.shape} score={s.shape}")
    # drop NaN scores
    ok = np.isfinite(s) & np.isfinite(y)
    return y[ok], s[ok]


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under precision–recall curve)."""
    y, s = _to_1d(y_true, y_score)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y, s = _to_1d(y_true, y_score)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(roc_auc_score(y, s))


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y, s = _to_1d(y_true, y_score)
    if y.size == 0:
        return float("nan")
    s = np.clip(s, 0.0, 1.0)
    return float(brier_score_loss(y, s))


def skill_vs_climatology(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_clim: np.ndarray,
    *,
    metric: str = "pr_auc",
) -> float:
    """
    Relative skill of y_pred over climatology y_clim.

    skill = (score_pred - score_clim) / max(score_clim, eps)
    Positive ⇒ better than climatology on the chosen metric.
    For Brier (lower is better), skill = (brier_clim - brier_pred) / max(brier_clim, eps).
    """
    fn = {"pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier}[metric]
    sp = fn(y_true, y_pred)
    sc = fn(y_true, y_clim)
    if not np.isfinite(sp) or not np.isfinite(sc):
        return float("nan")
    if metric == "brier":
        return float((sc - sp) / max(sc, 1e-12))
    return float((sp - sc) / max(sc, 1e-12))


def top_k_capture(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    """
    Fraction of real fires that fall in the top-k fraction of predicted-risk cells.

    k=0.10 → fraction of fire pixels among the highest 10% risk pixels.
    """
    y, s = _to_1d(y_true, y_score)
    if y.size == 0 or y.sum() == 0:
        return float("nan")
    k = float(k)
    if not (0.0 < k <= 1.0):
        raise ValueError("k must be in (0, 1]")
    n_top = max(1, int(np.ceil(k * y.size)))
    # top by score (stable: secondary by index via argpartition)
    idx = np.argpartition(-s, n_top - 1)[:n_top]
    return float(y[idx].sum() / y.sum())


def reliability_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 15,
) -> dict[str, np.ndarray]:
    """
    Reliability diagram data: per-bin predicted mean, observed frequency, counts.
    """
    y, s = _to_1d(y_true, y_score)
    s = np.clip(s, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(s, edges[1:-1], right=True)
    pred_mean = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = bin_ids == b
        counts[b] = int(m.sum())
        if counts[b] == 0:
            continue
        pred_mean[b] = float(s[m].mean())
        obs_freq[b] = float(y[m].mean())
    return {
        "bin_edges": edges,
        "pred_mean": pred_mean,
        "obs_freq": obs_freq,
        "counts": counts,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE = Σ (n_b / N) |obs_b - pred_b|."""
    rel = reliability_curve(y_true, y_score, n_bins=n_bins)
    counts = rel["counts"].astype(np.float64)
    n = counts.sum()
    if n == 0:
        return float("nan")
    pred = rel["pred_mean"]
    obs = rel["obs_freq"]
    ok = counts > 0
    ece = np.sum((counts[ok] / n) * np.abs(obs[ok] - pred[ok]))
    return float(ece)


def summarize(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    y_clim: np.ndarray | None = None,
    top_k: float = 0.10,
) -> dict[str, Any]:
    """One-shot metric dict for a model on a fold."""
    out: dict[str, Any] = {
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "brier": brier(y_true, y_score),
        "top10_capture": top_k_capture(y_true, y_score, k=top_k),
        "ece": expected_calibration_error(y_true, y_score),
        "n": int(np.asarray(y_true).size),
        "n_pos": int(np.asarray(y_true).sum()),
    }
    if y_clim is not None:
        out["skill_pr_vs_clim"] = skill_vs_climatology(
            y_true, y_score, y_clim, metric="pr_auc"
        )
    return out
