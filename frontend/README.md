# Prometheus website

React 19 + Vite + Leaflet. Visual system: [`Design.md`](./Design.md). How the
pages and APIs are wired: [`PLAN.md`](./PLAN.md).

```bash
# from the repo root, with the API already running on :8000
make ui
# http://localhost:5173   (proxies /api → 127.0.0.1:8000)
```

| Route | What it is |
|---|---|
| `/` | National map. Opens on **12 Apr 2026**. **Tomorrow / Next 7 days** on the left card and in the cell panel (same `?horizon=`). |
| `/predict` | What if — click a forest cell, move weather sliders. |
| `/district/:id` | One district’s mean/max and season line. |
| `/fires` | Satellite detections that already happened. |
| `/verify` | Did yesterday’s map catch today’s fires? |

Click a forest cell for a **chance (%)**, comparison chart, conditions, and
grouped driver shares — not SHAP slogans.

Install: `npm install`. Lint / production build: `npm run lint` and `npm run build`.
