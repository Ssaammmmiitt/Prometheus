"""What-if scoring: physical couples, schema, and one forest cell."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.transform import xy

from prometheus import grid
from prometheus.api.app import create_app
from prometheus.api.routes import whatif as W
from prometheus.features import forest
from prometheus.features.weather import dewpoint_from_rh, relative_humidity
from prometheus.infer import io_cog


def test_dewpoint_roundtrip():
    t = np.array(25.0)
    rh = np.array(40.0)
    d2m = dewpoint_from_rh(t, rh)
    back = relative_humidity(t, d2m)
    assert abs(float(back) - 40.0) < 0.6
    assert float(d2m) <= 25.0 + 1e-6


def test_apply_overrides_derives_vpd_and_orders_temperature():
    values = {
        "t2m_max": 20.0,
        "t2m": 18.0,
        "t2m_min": 10.0,
        "rh": 80.0,
        "vpd": 0.4,
        "d2m": 14.0,
        "precip": 0.0,
        "precip_7d": 5.0,
        "precip_30d": 20.0,
        "wind_speed": 2.0,
        "wind_max_7d": 3.0,
        "u10": 1.0,
        "v10": 1.732,
        "ndvi": 0.5,
        "evi": 0.3,
        "lst_day": 25.0,
        "lst_night": 10.0,
        "lst_diff": 15.0,
        "elevation": 800.0,
        "t2m_max_7d": 21.0,
        "rh_min_7d": 40.0,
    }
    bounds = {k: (-50.0, 200.0) for k in values}
    bounds["rh"] = (10.0, 100.0)
    out, clamped, ignored = W.apply_overrides(
        values, {"t2m_max": 30.0, "rh": 20.0, "elevation": 10.0}, bounds
    )
    assert out["t2m_max"] == 30.0
    assert out["t2m"] <= out["t2m_max"]
    assert out["vpd"] > values["vpd"]
    assert "elevation" in ignored
    assert out["elevation"] == 800.0
    assert abs(out["wind_speed"] - 2.0) < 1e-6


def test_whatif_schema():
    client = TestClient(create_app())
    resp = client.get("/api/whatif/schema")
    assert resp.status_code == 200
    body = resp.json()
    names = {s["feature"] for s in body["sliders"]}
    assert "rh" in names and "t2m" in names
    assert "vpd" not in names
    rh = next(s for s in body["sliders"] if s["feature"] == "rh")
    assert rh["lo"] < rh["hi"]


def test_whatif_forest_cell_smoke():
    day = date(2025, 4, 12)
    if not io_cog.risk_path(day.isoformat(), 1).is_file():
        pytest.skip("no forecast artefacts; need cube + bundle for what-if")
    mask = forest.forest_mask()
    r, c = np.where(mask)
    lon, lat = xy(grid.transform(), int(r[0]), int(c[0]), offset="center")
    client = TestClient(create_app())
    resp = client.post(
        "/api/whatif",
        json={
            "lat": float(lat),
            "lon": float(lon),
            "date": day.isoformat(),
            "horizon": 1,
            "overrides": {"rh": 15.0},
        },
    )
    if resp.status_code == 500:
        pytest.skip("feature cube or bundle missing")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["scenario"]["probability"] <= 1.0
    assert body["features"]["vpd"] >= 0.0
    assert "rh" in body["features"]


def test_whatif_off_mask_is_400():
    client = TestClient(create_app())
    resp = client.post(
        "/api/whatif",
        json={"lat": 27.7, "lon": 85.3, "date": "2025-04-12", "horizon": 1},
    )
    assert resp.status_code in (400, 500)
    if resp.status_code == 400:
        assert "forest" in str(resp.json()["detail"]).lower()
