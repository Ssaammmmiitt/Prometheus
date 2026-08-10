"""Leave-one-year-out evaluation for map-based risk scores."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from prometheus import grid
from prometheus.config import load_settings
from prometheus.eval import metrics as M
from prometheus.eval.baselines import (
    climatology_for_times,
    load_fire_cube,
    load_or_build_climatology,
    persistence_scores,
    year_indices,
)

PredictFn = Callable[[np.ndarray, pd.DatetimeIndex, int], np.ndarray]
# signature: (fire, times, test_year) -> score array same shape as fire for all T
# usually full-horizon scores precomputed; fold only slices year


def _masked_flat(
    y: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    time_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten year slice on mask only."""
    y_s = y[time_idx][:, mask]
    s_s = score[time_idx][:, mask]
    return y_s.ravel().astype(np.float64), s_s.ravel().astype(np.float64)


def evaluate_scores_loyo(
    fire: np.ndarray,
    times: pd.DatetimeIndex,
    scores: dict[str, np.ndarray],
    *,
    years: list[int] | None = None,
    top_k: float = 0.10,
    clim_name: str = "climatology",
) -> pd.DataFrame:
    """
    Leave-one-year-out style reporting: metrics per year (no refit for baselines).

    `scores` maps model name → (T,H,W) probability maps aligned with `fire`/`times`.
    """
    years = years if years is not None else list(load_settings().cv.years)
    mask = grid.nepal_mask()
    rows = []

    for year in years:
        t_idx = year_indices(times, year)
        if t_idx.size == 0:
            continue
        y_year = fire[t_idx][:, mask].ravel().astype(np.float64)
        base_rate = float(y_year.mean()) if y_year.size else float("nan")
        clim_flat = None
        if clim_name in scores:
            clim_flat = scores[clim_name][t_idx][:, mask].ravel().astype(np.float64)

        for name, sc in scores.items():
            s_flat = sc[t_idx][:, mask].ravel().astype(np.float64)
            summ = M.summarize(
                y_year,
                s_flat,
                y_clim=clim_flat if name != clim_name else None,
                top_k=top_k,
            )
            rows.append(
                {
                    "model": name,
                    "year": year,
                    "base_rate": base_rate,
                    **summ,
                }
            )
        print(f"  year {year}: base_rate={base_rate:.5f} n={y_year.size:,}", flush=True)
    return pd.DataFrame(rows)


def aggregate_loyo(per_year: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across years for display table."""
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "brier",
        "top10_capture",
        "ece",
        "skill_pr_vs_clim",
        "base_rate",
    ]
    rows = []
    for model, g in per_year.groupby("model"):
        rec: dict = {"model": model, "n_years": int(g["year"].nunique())}
        for col in metric_cols:
            if col not in g.columns:
                continue
            vals = g[col].astype(float)
            rec[f"{col}_mean"] = float(vals.mean())
            rec[f"{col}_std"] = float(vals.std(ddof=0))
        rows.append(rec)
    return pd.DataFrame(rows)


def run_baseline_loyo(
    *,
    force_clim: bool = False,
    lookback_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """
    Build climatology + persistence, evaluate leave-one-year-out (config cv.years).
    Returns (per_year, summary, scores_dict).
    """
    print("  loading fire cube …", flush=True)
    fire, times = load_fire_cube()
    print(f"  fire cube {fire.shape}", flush=True)
    print("  loading climatology …", flush=True)
    rates = load_or_build_climatology(force=force_clim)
    clim = climatology_for_times(rates, times)
    print("  computing persistence …", flush=True)
    pers = persistence_scores(fire, times, lookback_days=lookback_days)
    print("  scoring years …", flush=True)

    scores = {
        "climatology": clim,
        "persistence": pers,
    }
    top_k = float(load_settings().cv.top_k_fractions[-1])
    per_year = evaluate_scores_loyo(fire, times, scores, top_k=top_k)
    summary = aggregate_loyo(per_year)
    return per_year, summary, scores


def format_summary_table(summary: pd.DataFrame) -> str:
    """Pretty string matching the Day-3 done-when table."""
    lines = [
        f"{'model':<14} {'PR-AUC':>8} {'ROC-AUC':>8} {'Brier':>8} {'top10%-capture':>14}"
    ]
    order = ["climatology", "persistence"]
    models = list(summary["model"])
    for name in order + [m for m in models if m not in order]:
        row = summary[summary["model"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"{name:<14} "
            f"{r['pr_auc_mean']:8.4f} "
            f"{r['roc_auc_mean']:8.4f} "
            f"{r['brier_mean']:8.4f} "
            f"{r['top10_capture_mean']:14.4f}"
        )
    return "\n".join(lines)
