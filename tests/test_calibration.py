"""Tests for isotonic calibration, risk classes, bundle freezing, and inference."""

from __future__ import annotations

import json

import numpy as np
import pytest

from prometheus.models import calibrate as cal

pytest.importorskip("sklearn")


def _skewed_scores(n=200_000, base_rate=0.008, seed=0):
    """Scores that are well-ranked but wildly overconfident, like ours."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base_rate).astype(np.uint8)
    # Positives score higher, but both classes sit far above the true rate —
    # exactly what a 1:20 downsampled training set produces.
    raw = np.clip(rng.beta(2, 5, n) + 0.25 * y, 0, 1)
    return y, raw


def test_isotonic_maps_inflated_scores_onto_the_true_base_rate():
    y, raw = _skewed_scores()
    calibrator = cal.fit_isotonic(y, raw, fit_year=2025)

    assert raw.mean() > 20 * y.mean()  # the problem being solved
    calibrated = calibrator(raw)
    assert calibrated.mean() == pytest.approx(y.mean(), rel=0.05)
    assert calibrator.base_rate == pytest.approx(y.mean())
    assert calibrator.fit_year == 2025


def test_calibration_reduces_ece_by_orders_of_magnitude():
    y, raw = _skewed_scores()
    calibrator = cal.fit_isotonic(y, raw)
    report = cal.calibration_report(y, raw, calibrator(raw))

    assert report["ece_quantile_calibrated"] < report["ece_quantile_raw"] / 50
    # A monotone map cannot reorder, so discrimination must survive.
    assert report["pr_auc_calibrated"] == pytest.approx(report["pr_auc_raw"], abs=0.01)
    assert report["brier_calibrated"] < report["brier_raw"]


def test_calibrator_is_monotone_and_clipped_to_probabilities():
    y, raw = _skewed_scores()
    calibrator = cal.fit_isotonic(y, raw)
    probe = np.linspace(-0.5, 1.5, 500)
    out = calibrator(probe)

    assert np.all(np.diff(out) >= -1e-9)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert calibrator(np.zeros((3, 4))).shape == (3, 4)


def test_calibrator_round_trips_through_json(tmp_path):
    y, raw = _skewed_scores()
    calibrator = cal.fit_isotonic(y, raw, fit_year=2025)
    path = tmp_path / "cal.json"
    calibrator.save(path)
    restored = cal.Calibrator.load(path)

    np.testing.assert_allclose(calibrator(raw[:1000]), restored(raw[:1000]))
    assert json.loads(path.read_text())["fit_year"] == 2025


def test_fit_refuses_too_few_positives():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="positives"):
        cal.fit_isotonic(np.zeros(1000, dtype=np.uint8), rng.random(1000))


def test_risk_classes_partition_the_grid_by_quantile():
    rng = np.random.default_rng(1)
    p = rng.beta(1.5, 40, 500_000)
    quantiles = [0.0, 0.5, 0.75, 0.9, 0.95, 1.0]
    thresholds = cal.quantile_thresholds(p, quantiles)

    assert len(thresholds) == 4
    assert thresholds == sorted(thresholds)

    idx = cal.classify(p, thresholds)
    assert set(np.unique(idx)) == {0, 1, 2, 3, 4}
    shares = [float((idx == k).mean()) for k in range(5)]
    for got, want in zip(shares, [0.50, 0.25, 0.15, 0.05, 0.05]):
        assert got == pytest.approx(want, abs=0.01)


def test_classify_marks_nan_as_off_mask():
    p = np.array([[0.001, np.nan], [0.5, 0.02]])
    idx = cal.classify(p, [0.01, 0.05, 0.1, 0.2])
    assert idx[0, 1] == -1
    assert idx[0, 0] == 0 and idx[1, 0] == 4


def test_reliability_curve_bins_are_populated_and_ordered():
    y, raw = _skewed_scores()
    curve = cal.reliability_curve(y, raw, n_bins=10, strategy="quantile")
    assert np.all(curve["count"] > 0)
    assert np.all(np.diff(curve["mean_predicted"]) > 0)
    assert curve["mean_predicted"].size <= 10


def test_uniform_bins_understate_error_for_rare_events():
    """Why the report leads with equal-count bins rather than the textbook ones."""
    y, raw = _skewed_scores()
    uniform = cal.expected_calibration_error(y, raw, strategy="uniform")
    quantile = cal.expected_calibration_error(y, raw, strategy="quantile")
    assert uniform > 0 and quantile > 0
    assert abs(uniform - quantile) / max(quantile, 1e-9) < 5.0


def test_bundle_round_trips_with_calibrators_and_thresholds(tmp_path, monkeypatch):
    import lightgbm as lgb

    from prometheus.models import bundle as mb

    monkeypatch.setattr(mb, "bundles_root", lambda: tmp_path)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3)).astype(np.float32)
    y = (rng.random(2000) < 0.2).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 7},
        lgb.Dataset(x, label=y, feature_name=["a", "b", "c"]),
        num_boost_round=5,
    )
    model_path = tmp_path / "src_h1.txt"
    booster.save_model(str(model_path))

    ycal, raw = _skewed_scores(n=50_000)
    artifacts = mb.HorizonArtifacts(
        horizon=1,
        model_file=str(model_path),
        features=["a", "b", "c"],
        calibrator=cal.fit_isotonic(ycal, raw),
        risk_thresholds=[0.01, 0.05, 0.1, 0.2],
        metrics={"n_trees": 5},
    )
    original = mb.ModelBundle(
        version="v1", horizons={1: artifacts}, train_years=[2016, 2017],
        calibration_year=2025, test_year=2026,
        risk_class_names=["Low", "Moderate", "High", "VeryHigh", "Extreme"],
        risk_quantiles=[0.0, 0.5, 0.75, 0.9, 0.95, 1.0],
    )
    root = original.save()
    assert (root / "lgbm_h1.txt").is_file() and (root / mb.MANIFEST).is_file()

    loaded = mb.ModelBundle.load("v1")
    assert loaded.train_years == [2016, 2017]
    assert loaded.horizons[1].risk_thresholds == [0.01, 0.05, 0.1, 0.2]
    np.testing.assert_allclose(
        loaded.horizons[1].calibrator(raw[:500]), artifacts.calibrator(raw[:500])
    )
    # The booster must load from inside the bundle, not the original location.
    model_path.unlink()
    assert loaded.booster(1).num_trees() == booster.num_trees()


def test_next_version_increments(tmp_path, monkeypatch):
    from prometheus.models import bundle as mb

    monkeypatch.setattr(mb, "bundles_root", lambda: tmp_path)
    assert mb.next_version() == "v1"
    (tmp_path / "v1").mkdir()
    (tmp_path / "v7").mkdir()
    assert mb.next_version() == "v8"


def test_frozen_bundle_predicts_a_calibrated_grid():
    """End-to-end on the real frozen bundle, if one has been built."""
    from prometheus.models.bundle import bundles_root
    from prometheus.models.predict import RiskPredictor

    if not any(p.is_dir() for p in bundles_root().iterdir()):
        pytest.skip("no frozen bundle")

    predictor = RiskPredictor("latest")
    year = predictor.bundle.test_year
    surface = predictor.predict(f"{year}-04-01")

    assert surface.shape == (465, 912)
    finite = np.isfinite(surface)
    assert finite.sum() == 126_622  # exactly the forest mask
    assert 0.0 <= np.nanmin(surface) and np.nanmax(surface) <= 1.0
    # Calibrated output must sit near the true base rate, not the raw ~0.18.
    assert np.nanmean(surface) < 0.05

    classes = predictor.risk_classes(f"{year}-04-01")
    assert classes.shape == (465, 912)
    assert (classes == -1).sum() == (~finite).sum()
    assert set(np.unique(classes[finite])) <= {0, 1, 2, 3, 4}


def test_predictor_rejects_dates_outside_the_season():
    from prometheus.models.bundle import bundles_root
    from prometheus.models.predict import RiskPredictor

    if not any(p.is_dir() for p in bundles_root().iterdir()):
        pytest.skip("no frozen bundle")

    predictor = RiskPredictor("latest")
    with pytest.raises(KeyError, match="outside the modelled season"):
        predictor.predict(f"{predictor.bundle.test_year}-09-15")
