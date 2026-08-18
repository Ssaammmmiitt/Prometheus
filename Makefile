# Prometheus — Makefile

PYTHON ?= .prometheus-venv/bin/python
DATE   ?= 2026-04-12

.PHONY: forecast backfill-forecasts verify-forecasts api ui

forecast:
	$(PYTHON) -u scripts/forecast.py --date $(DATE)

backfill-forecasts:
	$(PYTHON) -u scripts/forecast.py --backfill 2024 2025 2026

verify-forecasts:
	$(PYTHON) -u scripts/forecast.py --verify 2024-01-01 2026-05-30

api:
	$(PYTHON) -u scripts/run_api.py

ui:
	cd frontend && npm run dev
