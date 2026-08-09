"""Unit tests for Day-3 metrics (fast, no full cube)."""

import numpy as np

from prometheus.eval import metrics as M


def test_pr_auc_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert M.pr_auc(y, s) > 0.9


def test_top_k_capture():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    s = np.array([0.99, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.98])
    # top 20% = 2 cells: scores 0.99 and 0.98 → both fires
    cap = M.top_k_capture(y, s, k=0.2)
    assert cap == 1.0


def test_skill_vs_clim():
    y = np.array([0, 0, 1, 1, 0, 1])
    clim = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    good = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.8])
    sk = M.skill_vs_climatology(y, good, clim, metric="pr_auc")
    assert sk > 0


def test_ece_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000).astype(float)
    s = rng.random(1000)
    e = M.expected_calibration_error(y, s)
    assert 0.0 <= e <= 1.0
