from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.transform import xy

from prometheus import grid
from prometheus.api.app import create_app
from prometheus.features import forest
from prometheus.infer import io_cog


def _first_forest_point() -> tuple[float, float]:
    mask = forest.forest_mask()
    r, c = np.where(mask)
    rr, cc = int(r[0]), int(c[0])
    lon, lat = xy(grid.transform(), rr, cc, offset="center")
    return float(lat), float(lon)


def test_verification_endpoint_smoke():
    v = io_cog.forecasts_dir() / "verification.csv"
    if not v.is_file():
        pytest.skip("no verification.csv found; run Day 13 forecast backfill first")
    client = TestClient(create_app())
    resp = client.get("/api/verification")
    assert resp.status_code == 200
    payload = resp.json()
    assert "summary" in payload
    assert isinstance(payload["rows"], list)


def test_tile_endpoint_smoke():
    day = "2025-04-12"
    path = io_cog.risk_path(day, 1)
    if not path.is_file():
        pytest.skip("no forecast risk COG found for 2025-04-12")
    client = TestClient(create_app())
    # Just pick any tile coordinate; we only assert PNG comes back.
    # z=0/0/0 is guaranteed to intersect the raster bbox.
    resp = client.get("/api/risk/tiles/0/0/0.png", params={"date": day, "horizon": 1})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert len(resp.content) > 100


def test_explain_endpoint_smoke():
    day = date(2025, 4, 12)
    pred = io_cog.risk_path(day.isoformat(), 1)
    if not pred.is_file():
        pytest.skip("no model outputs to explain; run make forecast first")
    lat, lon = _first_forest_point()
    client = TestClient(create_app())
    resp = client.get(
        "/api/explain",
        params={"lat": lat, "lon": lon, "date": day.isoformat(), "horizon": 1, "top": 6},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["horizon"] == 1
    assert len(payload["top"]) == 6
    assert all("feature" in it for it in payload["top"])
    assert 0.0 <= float(payload["probability"]) <= 1.0
    assert payload["compare"][0]["id"] == "here"
    assert payload["snapshot"]
    assert payload["drivers"]
    shares = [d["share"] for d in payload["drivers"]]
    assert all(0 <= s <= 1 for s in shares)
    assert abs(sum(shares) - 1.0) < 1e-6
    assert "chance of a satellite fire detection" in payload["headline"]


def test_health_and_forecast_catalogue():
    client = TestClient(create_app())
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    cat = client.get("/api/forecasts")
    assert cat.status_code == 200
    body = cat.json()
    assert "dates" in body
    if body["dates"]:
        assert body["default_date"] in body["dates"]
        years = {d[:4] for d in body["dates"]}
        if "2026" in years:
            assert "2026-04-12" in body["dates"] or body["default_date"].startswith("2026")


def test_tile_outside_bounds_is_transparent_png():
    day = "2025-04-12"
    if not io_cog.risk_path(day, 1).is_file():
        pytest.skip("no forecast risk COG found for 2025-04-12")
    client = TestClient(create_app())
    resp = client.get("/api/risk/tiles/8/128/128.png", params={"date": day, "horizon": 1})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")


def test_active_fires_empty_is_feature_collection():
    client = TestClient(create_app())
    resp = client.get("/api/fires/active", params={"as_of": "2010-01-01", "lookback_days": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert body["features"] == []


def test_districts_and_timeseries_smoke():
    day = "2025-04-12"
    if not io_cog.districts_path(day).is_file():
        pytest.skip("no district geojson for 2025-04-12")
    client = TestClient(create_app())
    geo = client.get("/api/districts", params={"date": day, "horizon": 1})
    assert geo.status_code == 200
    feats = geo.json()["features"]
    assert len(feats) == 77
    did = int(feats[0]["properties"]["district_id"])
    ts = client.get(
        f"/api/districts/{did}/timeseries",
        params={"horizon": 1, "start": "2025-04-01", "end": "2025-04-20"},
    )
    assert ts.status_code == 200
    payload = ts.json()
    assert payload["district_id"] == did
    assert len(payload["timeseries"]) >= 1

