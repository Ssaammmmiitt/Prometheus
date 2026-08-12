from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import COGReader

from prometheus.infer import io_cog

router_app = APIRouter()

# 1×1 transparent PNG so Leaflet never paints a broken-image icon for ocean tiles.
TRANSPARENT_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _colormap() -> list[tuple[tuple[float, float], tuple[int, int, int, int]]]:
    """Yellow→Orange→Red→Purple ramp for risk probability."""
    # Bins in probability space (since we always write calibrated 0..1 scores).
    # RGBA 255 alpha.
    return [
        ((0.0, 0.05), (255, 255, 204, 255)),  # pale yellow
        ((0.05, 0.1), (254, 204, 102, 255)),  # yellow-orange
        ((0.1, 0.2), (253, 141, 60, 255)),  # orange
        ((0.2, 0.4), (252, 78, 42, 255)),  # red
        ((0.4, 0.7), (189, 0, 38, 255)),  # dark red
        ((0.7, 1.0), (91, 33, 182, 255)),  # purple
    ]


COLORMAP = _colormap()


def _tile_png(path: Path, z: int, x: int, y: int) -> bytes:
    # Use a context manager so rasterio/GDAL state is cleaned up cleanly.
    with COGReader(str(path)) as reader:
        img = reader.tile(x, y, z)
        return img.render(img_format="PNG", colormap=COLORMAP)


def router_factory(root_fn):
    @router_app.get(
        "/risk/tiles/{z}/{x}/{y}.png",
        response_class=Response,
        responses={404: {"description": "missing forecast COG"}},
    )
    def tile(z: int, x: int, y: int, date: str, horizon: int = 1) -> Response:
        try:
            risk_path = io_cog.risk_path(date, horizon, root=root_fn())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not risk_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"missing forecast COG {risk_path.name}",
            )
        try:
            png = _tile_png(risk_path, z=z, x=x, y=y)
        except TileOutsideBounds:
            return Response(content=TRANSPARENT_PNG, media_type="image/png")
        return Response(content=png, media_type="image/png")

    return router_app


def router(root: Path | None = None) -> Any:
    if root is None:
        from prometheus.api.app import forecasts_root as root_fn  # type: ignore

        return router_factory(root_fn)
    return router_factory(lambda: root)


__all__ = ["router"]

