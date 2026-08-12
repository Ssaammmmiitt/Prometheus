#!/usr/bin/env python3
"""Day 14 — Run the FastAPI backend (local).

  python -u scripts/run_api.py

Then open:
  http://localhost:8000/docs
"""

from __future__ import annotations

import argparse

import uvicorn

from prometheus.api.app import create_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args(argv)

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

