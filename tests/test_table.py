"""Day 8: derived features must be causal, and the table must be balanced."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from prometheus.config import load_settings
from prometheus.features import derived
from prometheus.features.table import (
    META_COLUMNS,
    norm_stats_path,
    present_features,
    train_table_path,
)

# --------------------------------------------------------------------------
# Derived feature maths (no build artefacts needed)
# --------------------------------------------------------------------------


def test_rolling_sum_is_trailing_and_inclusive():
    arr = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    out = derived.rolling_sum(arr, 3).ravel()
    assert out[0] == 0
    assert out[1] == 0 + 1
    assert out[2] == 0 + 1 + 2
    assert out[5] == 3 + 4 + 5


def test_rolling_max_matches_brute_force():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(20, 3, 4)).astype(np.float32)
    window = 7
    out = derived.rolling_reduce(arr, window, "max")
    for t in range(arr.shape[0]):
        expected = arr[max(0, t - window + 1) : t + 1].max(axis=0)
        assert np.allclose(out[t], expected)


def test_rolling_min_matches_brute_force():
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(15, 2, 2)).astype(np.float32)
    out = derived.rolling_reduce(arr, 5, "min")
    for t in range(arr.shape[0]):
        expected = arr[max(0, t - 4) : t + 1].min(axis=0)
        assert np.allclose(out[t], expected)


def test_dry_counters_differ_and_reset_on_rain():
    precip = np.array([0.0, 0.0, 0.5, 0.0, 5.0, 0.0], dtype=np.float32).reshape(6, 1, 1)
    cdd, dsr = derived.dry_spell(precip)
    # 0.5 mm is below the 1 mm drying threshold but is measurable rain, so the
    # two counters must disagree at that step.
    assert cdd.ravel().tolist() == [1, 2, 3, 4, 0, 1]
    assert dsr.ravel().tolist() == [1, 2, 0, 1, 0, 1]


def test_horizon_labels_look_forward_only():
    fire = np.zeros((10, 1, 1), dtype=np.uint8)
    fire[5] = 1
    out = derived.horizon_labels(fire, [1, 3])
    h1 = out["label_h1"].ravel()
    assert h1[4] == 1, "day 4 must see the fire on day 5"
    assert h1[5] == 0, "a fire today is not a label for today"
    h3 = out["label_h3"].ravel()
    assert h3[2] == 1 and h3[3] == 1 and h3[4] == 1
    assert h3[1] == 0


def test_horizon_labels_mark_incomplete_lookahead_invalid():
    fire = np.zeros((10, 1, 1), dtype=np.uint8)
    out = derived.horizon_labels(fire, [7])
    valid = out["valid_h7"]
    assert valid[:3].all()
    assert not valid[-7:].any(), "days without a full 7-day window are not usable"


def test_fire_history_counts_past_seasons_only():
    shape = (1, 1)
    hist = derived.FireHistory(shape)

    first = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    fire = np.zeros((3, 1, 1), dtype=np.uint8)
    fire[1] = 1
    out = hist.process_year(first, fire)
    assert out["fires_1yr"].ravel().tolist() == [0, 1, 1]
    assert out["days_since_fire"].ravel()[0] == derived.NO_FIRE_SENTINEL
    assert out["days_since_fire"].ravel()[1] == 0, "a fire today means zero days since"
    assert out["days_since_fire"].ravel()[2] == 1

    second = [date(2021, 1, 1)]
    out2 = hist.process_year(second, np.zeros((1, 1, 1), dtype=np.uint8))
    assert out2["fires_1yr"].ravel()[0] == 1, "last season's single fire carries over"
    gap = (date(2021, 1, 1) - date(2020, 1, 2)).days
    assert out2["days_since_fire"].ravel()[0] == gap, "gap is calendar days, not cube steps"


def test_season_anomaly_excludes_the_target_year():
    anomaly = derived.SeasonAnomaly()
    dates = [date(2020, 1, 1), date(2020, 1, 2)]
    values = {
        2020: np.full((2, 1, 1), 10.0, dtype=np.float32),
        2021: np.full((2, 1, 1), 20.0, dtype=np.float32),
        2022: np.full((2, 1, 1), 30.0, dtype=np.float32),
    }
    for year, arr in values.items():
        anomaly.add_year(year, [d.replace(year=year) for d in dates], arr)

    # 2020 is scored against the mean of 2021 and 2022 only: 10 - 25 = -15.
    out = anomaly.anomaly(2020, [d.replace(year=2020) for d in dates], values[2020])
    assert np.allclose(out, -15.0)


def test_season_slot_keeps_leap_day_separate():
    assert derived.season_slot(date(2021, 3, 1)) == derived.season_slot(date(2020, 3, 1))
    assert derived.season_slot(date(2020, 2, 29)) == derived.season_slot(date(2000, 2, 29))


def test_day_of_year_encoding_is_bounded():
    sin, cos = derived.day_of_year_encoding([date(2021, 1, 1), date(2021, 5, 31)])
    assert np.all(np.abs(sin) <= 1) and np.all(np.abs(cos) <= 1)


# --------------------------------------------------------------------------
# The built table
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table():
    path = train_table_path()
    if not path.exists():
        pytest.skip(f"{path} not built — run scripts/build_train_table.py")
    import pandas as pd

    return pd.read_parquet(path)


def test_table_has_every_configured_feature(table):
    missing = set(load_settings().features.all_names) - set(table.columns)
    assert not missing, f"missing features: {sorted(missing)}"


def test_table_has_no_missing_values(table):
    bad = table.columns[table.isna().any()].tolist()
    assert not bad, f"NaNs in {bad}"


def test_negative_ratio_matches_config(table):
    ratio = load_settings().modeling.positive_negative_ratio
    pos = int(table["label_h1"].sum())
    neg = len(table) - pos
    assert pos > 0
    assert abs(neg / pos - ratio) < 0.5, f"got 1:{neg / pos:.2f}, want 1:{ratio}"


def test_sample_is_stratified_across_years_and_months(table):
    settings = load_settings()
    assert set(table["year"].unique()) == set(settings.years.all)
    for year in settings.years.all:
        months = set(table.loc[table["year"] == year, "month"].unique())
        assert months == set(settings.season.months), f"{year} missing months"


def test_every_stratum_holds_both_classes(table):
    grouped = table.groupby(["year", "month"])["label_h1"].agg(["mean", "size"])
    assert (grouped["mean"] > 0).all(), "a stratum has no positives"
    assert (grouped["mean"] < 1).all(), "a stratum has no negatives"


def test_rows_sit_inside_the_forest_mask(table):
    from prometheus.features import forest

    mask = forest.forest_mask()
    sample = table.sample(min(20_000, len(table)), random_state=0)
    assert mask[sample["row"].to_numpy(), sample["col"].to_numpy()].all()


def test_seven_day_label_covers_the_one_day_label(table):
    assert (table.loc[table["label_h1"] == 1, "label_h7"] == 1).all()


def test_meta_columns_are_not_used_as_features(table):
    features = set(present_features(table))
    assert not features & set(META_COLUMNS)
    assert not any(f.startswith("label_") for f in features)


def test_norm_stats_are_per_fold_and_versioned():
    path = norm_stats_path()
    if not path.exists():
        pytest.skip("normalisation stats not written yet")
    stats = json.loads(path.read_text())
    assert stats["version"] >= 1
    assert stats["scheme"] == "leave_one_year_out"
    years = load_settings().years.all
    assert set(stats["folds"]) == {str(y) for y in years}
    for fold in stats["folds"].values():
        for name, entry in fold.items():
            assert entry["std"] > 0, name


def test_norm_stats_exclude_the_held_out_year(table):
    """A fold's statistics must not have seen the year it will be tested on."""
    path = norm_stats_path()
    if not path.exists():
        pytest.skip("normalisation stats not written yet")
    stats = json.loads(path.read_text())
    held_out = load_settings().years.all[-1]
    feature = stats["features"][0]
    recorded = stats["folds"][str(held_out)][feature]["mean"]
    expected = float(table.loc[table["year"] != held_out, feature].mean())
    assert np.isclose(recorded, expected, rtol=1e-5)
