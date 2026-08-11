"""Tests for leave-one-year-out CV, family ablations, and region breakdown."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prometheus.eval import cv
from prometheus.features.table import feature_names

pytest.importorskip("lightgbm")


def test_families_partition_the_feature_set_without_overlap():
    families = cv.feature_families()
    seen: set[str] = set()
    for name, feats in families.items():
        overlap = seen & set(feats)
        assert not overlap, f"{name} overlaps earlier families on {overlap}"
        seen |= set(feats)


def test_every_feature_belongs_to_exactly_one_family_except_calendar():
    """Ablations only mean something if they cover the whole feature set."""
    covered = {f for feats in cv.feature_families().values() for f in feats}
    uncovered = set(feature_names()) - covered
    # Day-of-year encoding is calendar position, not an observed driver, so it is
    # deliberately held out of every family and never ablated.
    assert uncovered == {"doy_sin", "doy_cos"}


def test_dropping_a_family_removes_exactly_that_family():
    all_features = feature_names()
    for name, feats in cv.feature_families().items():
        kept = cv.ablation_features(all_features, name)
        present = set(all_features) & set(feats)
        assert len(kept) == len(all_features) - len(present)
        assert not set(kept) & present
        assert kept == [f for f in all_features if f in set(kept)]  # order preserved


def test_drop_none_keeps_everything_and_unknown_family_raises():
    all_features = feature_names()
    assert cv.ablation_features(all_features, None) == all_features
    with pytest.raises(KeyError):
        cv.ablation_features(all_features, "not_a_family")


def test_weather_ablation_also_removes_rolling_aggregates():
    """Keeping precip_30d while 'dropping weather' would not be an honest ablation."""
    kept = set(cv.ablation_features(feature_names(), "weather"))
    for leaked in ("precip_7d", "precip_30d", "t2m_max_7d", "rh_min_7d",
                   "wind_max_7d", "consecutive_dry_days", "days_since_rain"):
        assert leaked not in kept


def test_warmup_year_is_never_a_holdout_fold():
    from prometheus.config import load_settings

    years = sorted(load_settings().years.all)
    folds = [y for y in years if y != cv.HISTORY_WARMUP_YEAR]
    assert cv.HISTORY_WARMUP_YEAR == min(years)
    assert cv.HISTORY_WARMUP_YEAR not in folds
    assert len(folds) == len(years) - 1


def test_region_codes_cover_the_forest_mask():
    from prometheus.features import forest

    mask = forest.forest_mask()
    rows, cols = np.where(mask)
    codes = cv.region_codes(rows, cols)
    assert codes.shape == rows.shape
    assert set(np.unique(codes)) <= set(cv.REGION_NAMES) | {0}
    # No forest cell should fall outside the four physiographic belts.
    assert (codes == 0).sum() == 0


def test_fire_climatology_predates_the_modelling_period():
    """`fire_clim` dominates the ablation, so its independence has to be pinned."""
    import numpy as np

    from prometheus.config import load_settings

    path = load_settings().paths.resolve("cube") / "climatology_doy.npz"
    if not path.is_file():
        pytest.skip("climatology not built")
    years = np.load(path)["years"]
    assert years.max() < min(load_settings().years.all)


def test_fire_history_never_sees_the_day_it_predicts():
    """days_since_fire at day t may use detections through t, never t+1."""
    from datetime import date

    from prometheus.features import derived

    shape = (1, 1)
    fire = np.zeros((5, *shape), dtype=np.uint8)
    fire[3] = 1  # a single fire on day index 3
    dates = [date(2020, 3, 1 + i) for i in range(5)]

    out = derived.FireHistory(shape).process_year(dates, fire)
    gap = out["days_since_fire"][:, 0, 0]

    assert gap[2] == derived.NO_FIRE_SENTINEL  # day before: still nothing seen
    assert gap[3] == 0  # same day is observable when forecasting tomorrow
    assert gap[4] == 1

    counts = out["fires_1yr"][:, 0, 0]
    assert counts[2] == 0 and counts[3] == 1


def _fake_folds() -> list[dict]:
    folds = []
    for i, year in enumerate((2020, 2021)):
        base = 0.20 + 0.01 * i
        folds.append(
            {
                "year": year,
                "variants": {
                    "full": {"pr_auc": base, "top10_capture": 0.6, "n_features": 44,
                             "clim_pr_auc": 0.05, "persistence_pr_auc": 0.06,
                             "base_rate": 0.018, "skill_vs_clim": 3.0, "n_pos": 100},
                    "drop_weather": {"pr_auc": base - 0.02, "top10_capture": 0.5,
                                     "n_features": 25},
                },
                "regions": {
                    "Terai": {"pr_auc": base + 0.01, "n_pos": 50, "base_rate": 0.02,
                              "clim_pr_auc": 0.06, "top10_capture": 0.6,
                              "skill_vs_clim": 2.5},
                },
                "shap_mean_abs": {"days_since_fire": 0.5 + i * 0.1, "rh": 0.2},
                "region_shap": {"Terai": {"days_since_fire": 12.0, "rh": 4.0}},
            }
        )
    return folds


def test_summarise_reports_deltas_against_the_full_model():
    summary = cv.summarise(_fake_folds())
    ablation = summary["ablation"].set_index("variant")

    assert ablation.loc["full", "delta_mean"] == pytest.approx(0.0)
    assert ablation.loc["drop_weather", "delta_mean"] == pytest.approx(-0.02)
    assert ablation.loc["full", "pr_auc_mean"] == pytest.approx(0.205)
    assert ablation.loc["full", "pr_auc_std"] == pytest.approx(
        pd.Series([0.20, 0.21]).std()
    )


def test_summarise_builds_region_and_shap_tables():
    summary = cv.summarise(_fake_folds())

    assert list(summary["per_fold"]["year"]) == [2020, 2021]
    assert summary["regions"].iloc[0]["region"] == "Terai"
    assert summary["regions"].iloc[0]["n_pos"] == 100  # summed across folds

    shap_tbl = summary["shap_global"]
    assert shap_tbl.iloc[0]["feature"] == "days_since_fire"
    assert shap_tbl["share_pct"].sum() == pytest.approx(100.0)
    assert not summary["region_shap"].empty
