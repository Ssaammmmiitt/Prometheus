"""Evaluation metrics for wildfire risk maps."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter
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
    edges: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Reliability diagram data: per-bin predicted mean, observed frequency, counts.
    """
    y, s = _to_1d(y_true, y_score)
    s = np.clip(s, 0.0, 1.0)
    if edges is None:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        n_bins = len(edges) - 1
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

def reliability_by_class(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, np.ndarray]:
    """Reliability broken down by Prometheus risk classes: 0-5%, 5-10%, 10-20%, 20-40%, 40-100%"""
    edges = np.array([0.0, 0.05, 0.10, 0.20, 0.40, 1.0])
    return reliability_curve(y_true, y_score, edges=edges)


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

def fss(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray, threshold: float, window_size: int = 5) -> float:
    """Fractions Skill Score (FSS) over a 2D spatial grid."""
    if y_true.shape != y_score.shape:
        raise ValueError("Shape mismatch between true and predicted arrays.")
        
    # Apply threshold
    y = (y_true >= threshold).astype(float)
    s = (y_score >= threshold).astype(float)
    
    # Calculate fractional coverage
    y_frac = uniform_filter(y, size=window_size, mode='constant', cval=0.0)
    s_frac = uniform_filter(s, size=window_size, mode='constant', cval=0.0)
    
    # Only evaluate inside mask
    y_frac = y_frac[mask]
    s_frac = s_frac[mask]
    
    if y_frac.size == 0:
        return float("nan")
        
    mse = np.nanmean((y_frac - s_frac)**2)
    mse_ref = np.nanmean(y_frac**2 + s_frac**2)
    
    if mse_ref == 0:
        return float("nan")
        
    return float(1.0 - (mse / mse_ref))

def rev(y_true: np.ndarray, y_score: np.ndarray, c_ratio: float, threshold: float = None) -> float:
    """Relative Economic Value (REV)."""
    if threshold is None:
        threshold = c_ratio
        
    hits = np.sum((y_score >= threshold) & (y_true > 0))
    false_alarms = np.sum((y_score >= threshold) & (y_true == 0))
    misses = np.sum((y_score < threshold) & (y_true > 0))
    correct_negatives = np.sum((y_score < threshold) & (y_true == 0))
    
    N = hits + false_alarms + misses + correct_negatives
    if N == 0:
        return float("nan")
        
    base_rate = np.sum(y_true > 0) / N
    
    expense_clim = min(c_ratio, base_rate)
    expense_perf = base_rate * c_ratio
    expense_fcst = (hits + false_alarms) / N * c_ratio + (misses / N)
    
    if expense_clim - expense_perf == 0:
        return float("nan")
        
    return float((expense_clim - expense_fcst) / (expense_clim - expense_perf))
