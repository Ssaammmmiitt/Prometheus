"""Day 6-7: the feature cube must be one aligned, gap-free stack."""

from __future__ import annotations

import numpy as np
import pytest

from prometheus import grid
from prometheus.config import load_settings
from prometheus.features import forest
from prometheus.features.cube import (
    SPACE_CHUNK,
    TIME_CHUNK,
    cube_path,
    dynamic_variables,
    open_cube,
    season_dates,
)

MAX_NAN_FRACTION = 0.05


@pytest.fixture(scope="module")
def cube():
    path = cube_path()
    if not path.exists():
        pytest.skip(f"{path} not built yet — run scripts/build_feature_cube.py")
    return open_cube(path)


def test_grid_matches_config(cube):
    h, w = grid.shape()
    assert cube.sizes["y"] == h
    assert cube.sizes["x"] == w
    assert cube.attrs["crs"] == load_settings().grid.crs
    assert np.allclose(cube.attrs["transform"], list(grid.transform())[:6])


def test_every_layer_shares_the_grid(cube):
    h, w = grid.shape()
    for name, da in cube.data_vars.items():
        assert da.sizes["y"] == h, name
        assert da.sizes["x"] == w, name
        if "time" in da.dims:
            assert da.dims == ("time", "y", "x"), name
        else:
            assert da.dims == ("y", "x"), name


def test_coordinates_land_on_pixel_centres(cube):
    t = grid.transform()
    assert np.isclose(float(cube["x"].values[0]), t.c + 0.5 * t.a)
    assert np.isclose(float(cube["y"].values[0]), t.f + 0.5 * t.e)


def test_time_axis_is_the_fire_season(cube):
    expected = season_dates(list(cube.attrs["years"]))
    assert cube.sizes["time"] == len(expected)
    months = set(np.unique(cube["time"].dt.month.values).tolist())
    assert months == set(load_settings().season.months)


def test_all_dynamic_variables_present(cube):
    for name in dynamic_variables():
        assert name in cube.data_vars, name
        assert cube[name].dtype == np.float16, name


def test_chunking_matches_plan(cube):
    for name in dynamic_variables():
        chunks = cube[name].encoding.get("chunks") or cube[name].encoding.get("preferred_chunks")
        if chunks is None:
            continue
        sizes = tuple(chunks.values()) if isinstance(chunks, dict) else tuple(chunks)
        assert sizes == (TIME_CHUNK, SPACE_CHUNK, SPACE_CHUNK), (name, sizes)


def test_no_variable_exceeds_nan_budget_in_forest(cube):
    recorded = cube.attrs.get("nan_fraction_in_forest_mask", {})
    assert recorded, "build report missing NaN accounting"
    worst = max(recorded.values())
    assert worst <= MAX_NAN_FRACTION, f"worst NaN fraction {worst:.3%} > {MAX_NAN_FRACTION:.0%}"


def test_forest_mask_is_inside_nepal(cube):
    mask = forest.forest_mask()
    assert mask.shape == grid.shape()
    assert not (mask & ~grid.nepal_mask()).any()
    assert 0 < mask.sum() < grid.nepal_mask().sum()


def test_sampled_values_are_physical(cube):
    """Spot-check one April slice rather than reading the whole cube."""
    day = cube.sel(time=f"{cube.attrs['years'][-1]}-04-15")
    mask = grid.nepal_mask()

    t2m = day["t2m"].values.astype(np.float32)[mask]
    assert -50 < np.nanmin(t2m) and np.nanmax(t2m) < 50

    rh = day["rh"].values.astype(np.float32)[mask]
    assert 0 <= np.nanmin(rh) and np.nanmax(rh) <= 100.5

    precip = day["precip"].values.astype(np.float32)[mask]
    assert np.nanmin(precip) >= 0

    ndvi = day["ndvi"].values.astype(np.float32)[mask]
    assert -1 <= np.nanmin(ndvi) and np.nanmax(ndvi) <= 1

    pressure = day["surface_pressure"].values.astype(np.float32)[mask]
    assert 250 < np.nanmin(pressure) and np.nanmax(pressure) < 1100


def test_values_are_masked_outside_nepal(cube):
    day = cube.isel(time=0)
    outside = ~grid.nepal_mask()
    t2m = day["t2m"].values.astype(np.float32)
    assert np.isnan(t2m[outside]).all()


def test_lapse_correction_cools_the_high_peaks(cube):
    """High terrain must come out colder than the ~9 km field it came from."""
    from prometheus.features.warp import coarse_elevation, elevation_1km
    from prometheus.features.weather import read_month

    year = int(cube.attrs["years"][-1])
    _, coarse, transform = read_month(year, 4)
    fine = elevation_1km()
    ref = coarse_elevation(
        tuple(float(v) for v in transform[:6]),
        tuple(int(v) for v in coarse["t2m"].shape[1:]),
    )
    delta = ref - fine
    high = grid.nepal_mask() & (fine > 4000)
    assert delta[high].mean() < 0, "peaks should sit above the coarse-cell mean elevation"


def test_fire_labels_cover_the_same_seasons(cube):
    """Day 8 joins labels to features, so the two time axes must agree."""
    import xarray as xr

    path = load_settings().paths.resolve("cube") / "fire_daily.zarr"
    if not path.exists():
        pytest.skip("fire_daily.zarr not built")
    fire = xr.open_zarr(path, consolidated=False)
    missing = set(np.unique(cube["time"].dt.year.values).tolist()) - set(
        np.unique(fire["time"].dt.year.values).tolist()
    )
    assert not missing, (
        f"fire labels are missing years {sorted(missing)} — "
        "re-run scripts/build_fire_labels.py"
    )
