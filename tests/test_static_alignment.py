"""Static layers must sit on the canonical 1 km Nepal grid."""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.config import load_settings
from prometheus.grid import assert_aligned, shape


REQUIRED_STATIC = [
    "dist_road.tif",
    "dist_settlement.tif",
    "physio_regions.tif",
]


def test_grid_shape_from_config():
    assert shape() == (465, 912)


def test_nepal_mask_aligned_if_present():
    path = load_settings().paths.resolve("nepal_mask")
    if not path.is_file():
        pytest.skip(f"mask missing: {path}")
    assert_aligned(path)


def test_required_static_layers_aligned():
    static = load_settings().paths.resolve("static")
    missing = [n for n in REQUIRED_STATIC if not (static / n).is_file()]
    if missing:
        pytest.skip(
            "Local Day4–5 static not built yet: "
            + ", ".join(missing)
            + f" under {static}. Run scripts/build_local_static.py"
        )
    for name in REQUIRED_STATIC:
        assert_aligned(static / name)


def test_existing_elevation_slope_aligned_if_present():
    settings = load_settings()
    for key in ("elevation", "slope"):
        try:
            path = settings.paths.resolve(key)
        except Exception:
            continue
        if path.is_file():
            assert_aligned(path)
