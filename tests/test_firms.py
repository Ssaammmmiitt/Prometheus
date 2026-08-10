"""Day-2 fire label cube tests."""

from __future__ import annotations

import numpy as np
import pytest

from prometheus import grid
from prometheus.config import load_settings
from prometheus.data import firms as firms_mod


def test_season_dates_count():
    # non-leap: 31+28+31+30+31 = 151; leap Feb 29 → 152
    dates = firms_mod._season_dates(2020, 2020)
    assert len(dates) == 152
    dates2 = firms_mod._season_dates(2021, 2021)
    assert len(dates2) == 151


def test_clean_modis_confidence():
    import pandas as pd

    df = pd.DataFrame(
        {
            "latitude": [28.0, 28.1, 28.2],
            "longitude": [84.0, 84.1, 84.2],
            "acq_date": ["2021-04-01", "2021-04-01", "2021-04-01"],
            "confidence": [80, 30, 55],
            "type": [0, 0, 0],
            "satellite": ["T", "T", "A"],
            "collection": ["MODIS_SP", "MODIS_SP", "MODIS_SP"],
        }
    )
    out = firms_mod.clean_firms(df)
    assert len(out) == 2
    assert set(out["confidence"].astype(int)) == {80, 55}


def test_clean_viirs_confidence():
    import pandas as pd

    df = pd.DataFrame(
        {
            "latitude": [28.0, 28.1, 28.2],
            "longitude": [84.0, 84.1, 84.2],
            "acq_date": ["2021-04-01"] * 3,
            "confidence": ["nominal", "low", "high"],
            "type": [0, 0, 1],  # last is flare
            "satellite": ["N", "N", "N"],
            "collection": ["VIIRS_SNPP_SP"] * 3,
        }
    )
    out = firms_mod.clean_firms(df)
    # low dropped by conf, type=1 dropped → only nominal
    assert len(out) == 1
    assert out.iloc[0]["confidence"] == "nominal"


def test_rasterize_stays_in_mask():
    import pandas as pd

    # Point deep inside Nepal (approx Kathmandu valley)
    df = pd.DataFrame(
        {
            "latitude": [27.7],
            "longitude": [85.3],
            "acq_date": [pd.Timestamp("2021-04-15")],
            "confidence": [80],
            "type": [0],
            "satellite": ["T"],
            "collection": ["MODIS_SP"],
        }
    )
    cleaned = firms_mod.clean_firms(df)
    cube, times = firms_mod.rasterize_daily(cleaned, year_start=2021, year_end=2021)
    stats = firms_mod.assert_cube_alignment(cube)
    assert stats["outside_mask_fire_pixels"] == 0
    assert stats["total_fire_pixels"] >= 1  # dilate can add neighbors still in mask


def test_fire_cube_on_disk_if_built():
    """If Day-2 was run, assert cube integrity."""
    path = load_settings().paths.resolve("cube") / "fire_daily.zarr"
    if not path.exists():
        pytest.skip("fire_daily.zarr not built yet")
    import xarray as xr

    ds = xr.open_zarr(path)
    fire = ds["fire"].values.astype(np.uint8)
    stats = firms_mod.assert_cube_alignment(fire)
    assert stats["outside_mask_fire_pixels"] == 0
    assert fire.shape[1:] == grid.shape()
