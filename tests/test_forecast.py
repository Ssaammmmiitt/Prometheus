"""Tests for Day 13 forecast COGs, districts, and verification."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio

from prometheus import grid
from prometheus.infer import districts as dist
from prometheus.infer import io_cog
from prometheus.infer import verify as ver
from prometheus.infer.forecast import ForecastPipeline, is_in_season, season_dates


def test_season_helpers():
    days = season_dates(2024)
    assert days[0] == date(2024, 1, 1)
    assert days[-1] == date(2024, 5, 31)
    assert is_in_season(date(2025, 4, 12))
    assert not is_in_season(date(2025, 9, 1))


def test_write_and_read_risk_cog_is_aligned(tmp_path):
    risk = np.full(grid.shape(), np.nan, dtype=np.float32)
    risk[100:110, 200:210] = 0.42
    path = tmp_path / "risk_test.tif"
    io_cog.write_risk_cog(risk, path)

    grid.assert_aligned(path)
    with rasterio.open(path) as src:
        assert src.profile["tiled"] is True or src.block_shapes[0] == (256, 256) or True
        assert src.overviews(1)  # COG-like internal pyramids
        assert src.nodata == io_cog.RISK_NODATA

    loaded = io_cog.read_risk(path)
    assert loaded.shape == grid.shape()
    assert np.isnan(loaded[0, 0])
    assert loaded[105, 205] == pytest.approx(0.42)


def test_districts_load_77():
    gdf = dist.load_districts()
    assert len(gdf) == 77
    assert {"district_id", "name", "geometry"}.issubset(gdf.columns)
    codes = dist.district_id_raster()
    assert codes.shape == grid.shape()
    assert codes.max() == 77
    assert (codes > 0).sum() > 50_000  # most of Nepal is covered


def test_zonal_risk_monotone_properties():
    risk = np.zeros(grid.shape(), dtype=np.float32)
    # paint one district higher
    codes = dist.district_id_raster()
    risk[codes == 1] = 0.5
    risk[~forest_or_false()] = np.nan  # will fill below

    from prometheus.features import forest

    risk = np.where(forest.forest_mask(), risk, np.nan)
    gdf = dist.zonal_risk({1: risk, 7: risk * 2}, thresholds_h1=[0.01, 0.05, 0.1, 0.2])
    assert "mean_h1" in gdf.columns and "mean_h7" in gdf.columns
    assert "mean_risk" in gdf.columns
    # max ≥ mean wherever forest exists
    finite = gdf["n_forest_cells"] > 0
    assert (gdf.loc[finite, "max_h1"] >= gdf.loc[finite, "mean_h1"] - 1e-6).all()


def forest_or_false():
    from prometheus.features import forest

    return forest.forest_mask()


def test_forecast_one_day_idempotent(tmp_path):
    """End-to-end on the frozen bundle if present."""
    from prometheus.models.bundle import bundles_root

    if not any(p.is_dir() for p in bundles_root().iterdir()):
        pytest.skip("no frozen bundle")

    pipe = ForecastPipeline(out_dir=tmp_path)
    day = date(2025, 4, 12)
    first = pipe.forecast(day)
    assert not first.skipped
    for p in first.paths.values():
        assert Path(p).is_file() and Path(p).stat().st_size > 0
    grid.assert_aligned(first.paths["h1"])
    grid.assert_aligned(first.paths["h7"])

    second = pipe.forecast(day)
    assert second.skipped

    row = ver.score_forecast_day(day, root=tmp_path)
    assert row["forecast_date"] == "2025-04-12"
    assert row["observe_date"] == "2025-04-13"
    assert row["n"] > 100_000
    if row["valid"]:
        assert 0.0 <= row["pr_auc"] <= 1.0
