"""Day-1 scaffolding checks."""

from prometheus import grid
from prometheus.config import cfg, load_settings


def test_years_and_season():
    assert cfg.years == list(range(2016, 2027))
    assert cfg.season_months == [1, 2, 3, 4, 5]


def test_settings_load():
    s = load_settings()
    assert s.grid.height == 465
    assert s.grid.width == 912
    assert s.grid.crs == "EPSG:4326"
    assert s.years.train_end == 2026
    assert len(s.cv.years) == 11
    assert s.cv.years[-1] == 2026


def test_grid_shape():
    assert grid.shape() == (465, 912)
    t = grid.transform()
    assert abs(t.a - 0.008983152841195215) < 1e-12
    assert abs(t.c - 80.01294235652578) < 1e-9
    assert abs(t.f - 30.515770201540146) < 1e-9


def test_mask_aligned_if_present():
    path = grid.mask_path()
    if path.is_file():
        grid.assert_aligned(path)
        assert grid.n_valid_pixels() == 168064
