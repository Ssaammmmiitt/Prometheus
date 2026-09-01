"""Explain-panel helpers: grouping, snapshot figures, comparison copy."""

from __future__ import annotations

import numpy as np
import pytest

from prometheus.api.routes.explain import (
    compare_rows,
    format_snapshot_item,
    grouped_drivers,
    headline,
    snapshot_from_values,
)
from prometheus.features.derived import NO_FIRE_SENTINEL


def test_grouped_drivers_merges_fire_history():
    names = ["days_since_fire", "fire_clim", "fires_1yr", "rh", "t2m_max"]
    shap = np.array([0.4, 0.2, 0.1, -0.2, 0.05], dtype=np.float64)
    rows = grouped_drivers(names, shap, limit=6)
    by_key = {row["key"]: row for row in rows}
    assert by_key["fire"]["shap"] == pytest.approx(0.7)
    assert by_key["fire"]["direction"] == "up"
    assert by_key["moisture"]["direction"] == "down"
    assert sum(row["share"] for row in rows) == pytest.approx(1.0)
    assert rows[0]["key"] == "fire"


def test_snapshot_formats_units_and_sentinel():
    none = format_snapshot_item("days_since_fire", NO_FIRE_SENTINEL)
    assert none is not None
    assert none["display"] == "None this season"
    rh = format_snapshot_item("rh", 28.4)
    assert rh is not None
    assert rh["display"] == "28"
    assert rh["unit"] == "%"
    clim = format_snapshot_item("fire_clim", 0.0142)
    assert clim is not None
    assert clim["display"] == "1.42"
    tree = format_snapshot_item("tree_frac", 0.62)
    assert tree is not None
    assert tree["display"] == "62"
    rows = snapshot_from_values(
        {
            "t2m_max": 31.24,
            "rh": 28.0,
            "days_since_fire": NO_FIRE_SENTINEL,
            "elevation": 1842.6,
        }
    )
    keys = [row["key"] for row in rows]
    assert keys[:2] == ["t2m_max", "rh"]
    assert any(row["display"] == "None this season" for row in rows)


def test_headline_and_compare_rows():
    text = headline(0.024, 0.008, 82.0, 1)
    assert "2.40%" in text
    assert "3.0×" in text
    assert "82%" in text
    assert "tomorrow" in text
    rows = compare_rows(
        0.024,
        0.008,
        {"mean": 0.01, "percentile": 82.0},
        {"name": "Kaski", "mean": 0.012},
    )
    ids = [row["id"] for row in rows]
    assert ids == ["here", "district", "country", "typical"]
    assert rows[1]["label"].startswith("Kaski")
