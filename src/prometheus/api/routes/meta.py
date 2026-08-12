"""Catalogue endpoints the UI needs before it paints a map."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

router_app = APIRouter()

DEFAULT_DATE = "2025-04-12"


def _default(dates: list[str]) -> str | None:
    if DEFAULT_DATE in dates:
        return DEFAULT_DATE
    return dates[-1] if dates else None


def available_dates(root: Path, horizon: int = 1) -> list[str]:
    dates = []
    for path in root.glob(f"risk_*_h{horizon}.tif"):
        # risk_YYYY-MM-DD_h1.tif
        stem = path.stem  # risk_2025-04-12_h1
        parts = stem.split("_")
        if len(parts) >= 3:
            dates.append(parts[1])
    return sorted(set(dates))


def router_factory(root_fn):
    @router_app.get("/health")
    def health() -> dict[str, Any]:
        root = Path(root_fn())
        dates = available_dates(root)
        return {
            "ok": True,
            "service": "prometheus",
            "n_forecasts": len(dates),
            "default_date": _default(dates),
        }

    @router_app.get("/forecasts")
    def forecasts() -> dict[str, Any]:
        root = Path(root_fn())
        dates = available_dates(root)
        years = sorted({d[:4] for d in dates})
        return {
            "dates": dates,
            "years": years,
            "default_date": _default(dates),
            "horizons": [1, 7],
        }

    return router_app


def router(root: Path | None = None) -> Any:
    if root is None:
        from prometheus.api.app import forecasts_root as root_fn  # type: ignore

        return router_factory(root_fn)
    return router_factory(lambda: root)


__all__ = ["DEFAULT_DATE", "available_dates", "router"]
