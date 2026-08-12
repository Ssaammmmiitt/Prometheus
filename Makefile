# Prometheus — Makefile

PYTHON ?= .prometheus-venv/bin/python
DATE   ?= 2025-04-12

.PHONY: forecast backfill-forecasts verify-forecasts api ui

forecast:
	$(PYTHON) -u scripts/forecast.py --date $(DATE)

# App history. Add 2026 after the cube has that season:
#   $(PYTHON) -u scripts/forecast.py --backfill 2026
backfill-forecasts:
	$(PYTHON) -u scripts/forecast.py --backfill 2024 2025

verify-forecasts:
	$(PYTHON) -u scripts/forecast.py --verify 2024-01-01 2025-05-30

api:
	$(PYTHON) -u scripts/run_api.py

ui:
	cd frontend && npm run dev
