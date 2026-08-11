"""Tests for the LightGBM fold trainer and full-grid scoring plumbing."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from prometheus.features import derived
from prometheus.models import lgbm

lgb = pytest.importorskip("lightgbm")


def _fake_table(years=(2016, 2017, 2018, 2019), n=4000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for year in years:
        x = rng.normal(size=(n, 3)).astype(np.float32)
        logit = 1.5 * x[:, 0] - 1.0 * x[:, 1] - 3.0
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.uint8)
        frames.append(
            pd.DataFrame(
                {
                    "t2m_max": x[:, 0],
                    "rh": x[:, 1],
                    "elevation": x[:, 2],
                    "label_h1": y,
                    "year": np.full(n, year, dtype=np.int16),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_split_fold_excludes_holdout_and_reserves_a_whole_season():
    table = _fake_table()
    fit, valid = lgbm.split_fold(table, 2019)
    assert 2019 not in set(fit["year"]) and 2019 not in set(valid["year"])
    assert valid["year"].nunique() == 1
    # Early stopping must not see rows that also appear in the fit set.
    assert set(fit["year"]).isdisjoint(set(valid["year"]))


def test_split_fold_needs_enough_years():
    with pytest.raises(ValueError):
        lgbm.split_fold(_fake_table(years=(2016,)), 2016)


def test_train_fold_learns_and_reports_scale_pos_weight(monkeypatch):
    table = _fake_table()
    booster, result = lgbm.train_fold(table, 2019, num_boost_round=60)

    fit, _ = lgbm.split_fold(table, 2019)
    expected = (len(fit) - fit["label_h1"].sum()) / fit["label_h1"].sum()
    assert result.scale_pos_weight == pytest.approx(expected)

    assert result.best_iteration <= 60
    assert result.valid_pr_auc > table["label_h1"].mean()  # beats the base rate
    assert list(booster.feature_name()) == ["t2m_max", "rh", "elevation"]


def test_param_grid_is_distinct_and_covers_the_four_knobs():
    configs = lgbm.sample_param_grid(24)
    assert len(configs) == 24
    assert len({tuple(sorted(c.items())) for c in configs}) == 24
    for key in ("num_leaves", "min_data_in_leaf", "learning_rate", "feature_fraction"):
        assert all(key in c for c in configs)
        assert len({c[key] for c in configs}) > 1


def test_importance_pools_gain_across_collinear_twins():
    table = _fake_table()
    booster, _ = lgbm.train_fold(table, 2019, num_boost_round=40)
    frame = lgbm.feature_importance(booster).set_index("feature")

    assert frame["gain_pct"].sum() == pytest.approx(100.0, abs=1e-6)
    # t2m_max has no twin present here; elevation's partner is surface_pressure,
    # which is absent, so its pooled share falls back to its own.
    assert frame.loc["elevation", "pair_gain_pct"] == pytest.approx(
        frame.loc["elevation", "gain_pct"]
    )
    assert pd.isna(frame.loc["rh", "collinear_with"])


def test_fire_history_advance_matches_process_year_state():
    """Replaying history for inference must land on the same state as training."""
    rng = np.random.default_rng(3)
    shape = (6, 5)
    fire = (rng.random((10, *shape)) < 0.2).astype(np.uint8)
    dates = [date(2018, 3, 1 + i) for i in range(10)]

    a, b = derived.FireHistory(shape), derived.FireHistory(shape)
    a.process_year(dates, fire)
    b.advance(dates, fire)

    np.testing.assert_array_equal(a.last_fire_ord, b.last_fire_ord)
    assert len(a.season_totals) == len(b.season_totals)
    np.testing.assert_array_equal(a.season_totals[-1], b.season_totals[-1])
