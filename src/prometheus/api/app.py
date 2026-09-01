"""Day 14 — FastAPI backend.

Routes are designed to be thin wrappers over the existing Day 13 artefacts:
COG risk rasters + `districts_{date}.geojson` + `verification.csv`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prometheus.api.routes import districts, explain, fires, meta, risk_tiles, verification, whatif


def forecasts_root() -> Path:
    """Override in tests via `PROMETHEUS_FORECASTS_ROOT`."""
    from prometheus.infer import io_cog

    env = os.getenv("PROMETHEUS_FORECASTS_ROOT")
    return Path(env) if env else io_cog.forecasts_dir()


@lru_cache(maxsize=1)
def _meta_router() -> Any:
    return meta.router()


@lru_cache(maxsize=1)
def _risk_tiles_router() -> Any:
    return risk_tiles.router()


@lru_cache(maxsize=1)
def _districts_router() -> Any:
    return districts.router()


@lru_cache(maxsize=1)
def _fires_router() -> Any:
    return fires.router()


@lru_cache(maxsize=1)
def _verification_router() -> Any:
    return verification.router()


@lru_cache(maxsize=1)
def _explain_router() -> Any:
    return explain.router()


@lru_cache(maxsize=1)
def _whatif_router() -> Any:
    return whatif.router()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Prometheus API",
        version="0.1",
        description="Forecast tiles and district summaries for Nepal wildfire risk.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(_meta_router(), prefix="/api")
    app.include_router(_risk_tiles_router(), prefix="/api")
    app.include_router(_districts_router(), prefix="/api")
    app.include_router(_fires_router(), prefix="/api")
    app.include_router(_verification_router(), prefix="/api")
    app.include_router(_explain_router(), prefix="/api")
    app.include_router(_whatif_router(), prefix="/api")

    # Serve static frontend if it exists
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException
    
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")
        
        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            # If the path looks like a file, try to serve it directly
            potential_file = frontend_dist / full_path
            if potential_file.is_file():
                return FileResponse(potential_file)
            # Otherwise, fall back to index.html for React Router
            return FileResponse(frontend_dist / "index.html")
            
    return app


app = create_app()

