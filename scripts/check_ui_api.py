#!/usr/bin/env python3
"""Simulate the Day 15 UI call sequence against the FastAPI app."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from rasterio.transform import xy

from prometheus import grid
from prometheus.api.app import create_app
from prometheus.features import forest


def _forest_point() -> tuple[float, float]:
    mask = forest.forest_mask()
    r, c = np.where(mask)
    lon, lat = xy(grid.transform(), int(r[0]), int(c[0]), offset="center")
    return float(lat), float(lon)


def main() -> int:
    client = TestClient(create_app())
    checks = []

    health = client.get("/api/health")
    checks.append(("health", health.status_code == 200 and health.json()["ok"]))

    cat = client.get("/api/forecasts")
    dates = cat.json().get("dates", [])
    checks.append(("forecasts", cat.status_code == 200 and "2025-04-12" in dates))

    day = "2025-04-12"
    tile = client.get("/api/risk/tiles/7/90/55.png", params={"date": day, "horizon": 1})
    png = tile.headers["content-type"].startswith("image/png")
    checks.append(("tile h1", tile.status_code == 200 and png))

    oob = client.get("/api/risk/tiles/8/128/128.png", params={"date": day, "horizon": 1})
    checks.append(("tile oob", oob.status_code == 200))

    geo = client.get("/api/districts", params={"date": day, "horizon": 1})
    feats = geo.json().get("features", []) if geo.status_code == 200 else []
    checks.append(("districts 77", geo.status_code == 200 and len(feats) == 77))

    did = int(feats[0]["properties"]["district_id"]) if feats else 1
    ts = client.get(
        f"/api/districts/{did}/timeseries",
        params={"horizon": 1, "start": "2025-04-01", "end": "2025-04-20"},
    )
    n_ts = len(ts.json().get("timeseries", [])) if ts.status_code == 200 else 0
    checks.append(("timeseries", ts.status_code == 200 and n_ts > 0))

    fires = client.get("/api/fires/active", params={"as_of": day, "lookback_days": 2})
    is_fc = fires.status_code == 200 and fires.json()["type"] == "FeatureCollection"
    checks.append(("fires", is_fc))

    empty = client.get("/api/fires/active", params={"as_of": "2010-01-01", "lookback_days": 1})
    checks.append(
        ("fires empty", empty.status_code == 200 and empty.json()["features"] == [])
    )

    ver = client.get("/api/verification", params={"start": "2024-01-01", "end": "2025-05-30"})
    checks.append(("verification", ver.status_code == 200 and "summary" in ver.json()))

    lat, lon = _forest_point()
    expl = client.get(
        "/api/explain", params={"lat": lat, "lon": lon, "date": day, "horizon": 1}
    )
    ok_expl = expl.status_code == 200 and len(expl.json()["top"]) == 6
    checks.append(("explain forest", ok_expl))

    off = client.get(
        "/api/explain", params={"lat": 27.7, "lon": 85.3, "date": day, "horizon": 1}
    )
    checks.append(("explain not 500", off.status_code in (200, 400)))

    failed = [(n, ok) for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'ok' if ok else 'FAIL':<4} {name}")
    if failed:
        print(f"{len(failed)} failed")
        return 1
    print(f"{len(checks)} UI API checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
