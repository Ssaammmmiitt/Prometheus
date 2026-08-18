# Prometheus frontend — Day 15 plan (implemented)

Build the operational map app inside this folder. Reuse the existing React + Vite + Tailwind + Leaflet stack. Visual language comes from [`Design.md`](./Design.md) (**Arcade Night**). Dark is the default; light is a switchable inversion of the same tokens, not a second design system.

**Shipped:** the app loads real forecasts from the API, the date scrubber animates **Jan–May 2024–2026** (opens on **2026-04-12**), light/dark toggles without restyling by hand, **What if** lives at `/predict`, and the cell panel is a statistical explain drawer. **Tomorrow / Next 7 days** appears on the left layer card **and** in that panel; both call `setHorizon` in `ForecastContext` so the map, URL, and stats stay in lockstep.

The sections below were the original build plan. Where they disagree with the shipped app, the **shipped** behaviour wins.

---

## 0. What already exists (keep vs throw)

| Keep | Throw |
|---|---|
| Vite + React 19 + Tailwind 4 | Mock `generatePatches` / `generateHistoricalFires` |
| `leaflet` + `react-leaflet` | `public/patch_grid.geojson` as the risk layer |
| `react-router-dom` (unused today — wire it) | Slate/blue Tailwind (`bg-slate-950`, `rounded-xl`) |
| `recharts`, `lucide-react`, `date-fns`, `clsx` | Red–green probability ramp |
| Nepal bounds / center constants | Fake model versions, 16-day window, threshold slider as the product |

The current `src/page.jsx` is a single-file prototype. Split it. Do not restyle the prototype in place.

---

## 1. Design system — Arcade Night, plus a light invert

[`Design.md`](./Design.md) is dark-only and forbids inventing colors. Light mode is an explicit product requirement, so it is a **role inversion of the same named tokens**, not new hues.

### 1.1 Token contract (CSS variables, once)

Declare in `src/styles/tokens.css`. Components use `var(--color-*)` / Tailwind theme mapping — never raw hex in JSX.

| Role | Dark (Arcade Night) | Light (invert) |
|---|---|---|
| `--color-surface` | `#0a0b08` | `#eef0e6` |
| `--color-lift` | `#11120f` | `#eef0e6` with hairline (cards sit on surface) |
| `--color-text` | `#eef0e6` | `#0a0b08` |
| `--color-muted` | `#8a8f80` | `#8a8f80` |
| `--color-accent` | `#c8ff3a` | `#c2410c` (shipped ember; Design.md’s lime-on-light was not used) |
| `--color-accent-fg` | `#0a0b08` | `#0a0b08` |
| `--color-hairline` | `rgba(238,240,230,0.06)` | `rgba(17,18,15,0.10)` |
| `--color-hairline-strong` | `rgba(238,240,230,0.18)` | `rgba(17,18,15,0.18)` |
| `--color-live` | `#ff3b3b` | `#ff3b3b` (live indicator only) |
| `--color-accent-12` | `rgba(200,255,58,0.12)` | `rgba(200,255,58,0.18)` |

**Unchanged across themes:** Big Shoulders Display + JetBrains Mono, radius `0 / 0 / 2px / pill`, volt lime as the only accent, knife-edge corners, no glows, no second accent.

**Map basemap:** Carto Dark (`dark_all`) in dark mode, Carto Light (`light_all`) in light mode. Risk tiles keep the yellow→orange→red→purple ramp from the API (colourblind-safe; do not recolor in the client).

### 1.2 Typography and chrome

- Display font: headlines, brand wordmark `PROMETHEUS`, score numerals.
- Mono: every UI label, uppercase, `letter-spacing: 0.10em`.
- Buttons: only the four recipes in Design.md (primary / secondary / outline / ghost). One primary CTA per screen.
- Cards: hairline, no drop shadow. Featured cards may use `radius.lg` (2px).
- Tabs: underline, 2px accent, no pills.
- Charts: Recharts, flat, no gridlines, one highlighted series in volt lime, fill opacity 0.18.

### 1.3 Motion (dashboard override)

Design.md Pro tokens include bounce + rotate. That fights a map UI. Follow the **hard constraint** in Design.md §4: transitions `≤200ms`, no rotation on panels, buttons, or map chrome. Theme switch: swap `data-theme` on `<html>` in one frame; persist `localStorage.theme` (`dark` \| `light`). Default **dark**.

### 1.4 Theme wiring

```
<html data-theme="dark">   <!-- or light -->
```

- `ThemeProvider` in `src/theme/ThemeProvider.jsx`
- Toggle in the app header (Sun / Moon from `lucide-react`)
- Leaflet tile URL swaps with theme; overlays do not

---

## 2. Pages (five views)

Routes via `react-router-dom`. Shared shell: top bar (wordmark, nav tabs, date, theme toggle). Horizon is **not** only in the header — it is `?horizon=1|7` in `ForecastContext`, toggled from the map’s left card and from the right-hand cell panel.

| Route | View | Notes |
|---|---|---|
| `/` | National risk map | Default date **2026-04-12** |
| `/predict` | What-if weather sandbox | Click a forest cell, then sliders |
| `/district/:id` | District drill-down | Mean/max + season timeseries |
| `/fires` | Fire explorer | FIRMS points (history, not the forecast) |
| `/verify` | Verification | Accuracy; base rate always shown |

Season constraint: forecasts exist for **Jan–May 2024, 2025, and 2026**. Date picker clamps to in-season days that have artefacts; 404s become an empty state, not a crash.

### 2.1 `/` — National risk map

**Layout:** full-viewport Leaflet. Overlay chrome, not a document page.

| Chrome | Behavior |
|---|---|
| Date scrubber | Range input + play/pause. Year tabs newest-first. Animating play is the Day 15 acceptance test. |
| Horizon toggle | Tabs: **Tomorrow** / **Next 7 days**. Query `horizon=1\|7`. Same control is repeated in the cell panel; both write `setHorizon`. |
| Layer toggles | Risk tiles on/off + opacity. District outlines on/off. Active fires on/off. |
| Legend | Yellow→orange→red→purple bins matching the tile colormap (not red–green). |
| Stats strip | Count of Extreme / Very High districts from `/api/districts`. |
| Click map | `GET /api/explain?lat=&lon=&date=&horizon=` → right drawer: **calibrated %**, class badge, comparison chart (this cell / district / Nepal / typical), condition snapshot, grouped driver shares. Off-forest → muted empty state. |
| Click district | Same explain drawer at the click point (not an immediate navigate). **See all of …** keeps `date` + `horizon` and opens `/district/:id`. |

**APIs**

```
GET /api/risk/tiles/{z}/{x}/{y}.png?date=YYYY-MM-DD&horizon=1|7
GET /api/districts?date=YYYY-MM-DD&horizon=1|7
GET /api/fires/active?as_of=YYYY-MM-DD&lookback_days=2
GET /api/explain?lat=&lon=&date=&horizon=&top=6
```

`/api/explain` JSON includes `probability`, `risk_class_name`, `base_rate`, `vs_country`, `district`, `compare`, `snapshot`, `drivers`, `headline`, and `top` (raw SHAP rows for back-compat).

**Leaflet risk layer**

```js
L.tileLayer("/api/risk/tiles/{z}/{x}/{y}.png?date={date}&horizon={horizon}", {
  opacity: 0.65,
  maxZoom: 12,
  bounds: NEPAL_BOUNDS,
})
```

Rebuild the layer when `date` or `horizon` changes (new `TileLayer` key). Do not fetch rasters in JS.

**District GeoJSON** — `L.geoJSON` from `/api/districts`. Fill by `risk_class` (0–4) using the same yellow→purple family at low opacity; stroke hairline. Popup: name, `mean_h{horizon}`, class name.

### 2.2 `/district/:id` — District drill-down

**Layout:** map (district highlighted, rest dimmed) + right column.

| Block | Source |
|---|---|
| Title, class badge, mean/max probability | Feature from `/api/districts?date&horizon` |
| Time series chart | `/api/districts/{id}/timeseries?horizon&start&end` |
| Horizon tabs | Same **Tomorrow / Next 7 days** as the map (`setHorizon`) |
| Back to national | `/` with query preserved |

Chart: Recharts line, volt lime, 18% fill, no gridlines, mono axis labels. Highlight the selected date (single-point strategy from Design.md).

`start` / `end` default to the season of the selected date (e.g. `2026-01-01` … `2026-05-31`).

### 2.3 `/fires` — Fire explorer

Not a second risk map. FIRMS detections as points.

| Control | API |
|---|---|
| `as_of` date | `/api/fires/active?as_of=&lookback_days=` |
| Lookback 1–7 days | same |
| Optional risk underlay | same tile URL as `/` |

Points: small squares (not round pins — system is sharp). Live-red only if `as_of` is “today” *and* detections exist; otherwise bone/muted. This page is historical for 2024–2026 cube dates.

Empty / out-of-cube dates: copy that detections exist only where the fire cube has that day — do not invent points.

### 2.4 `/verify` — Verification

No map required (optional small Nepal inset later). This is the honesty page.

| Block | Source |
|---|---|
| Summary scores | `summary` from `GET /api/verification?start&end` |
| Daily table | `rows` — `forecast_date`, `n_pos`, `base_rate`, `pr_auc`, `top10_capture`, `brier` |
| Sparkline | `pr_auc` over the range; one highlighted day |

Copy that must appear: daily PR-AUC is noisier than season-level LOYO (Day 10 mean **0.1548**). Always show **base rate** next to a metric. Quiet days (`valid: false`) stay in the table as em-dash, not zero.

Default range: `2024-01-01` … `2026-05-30`.

---

### 2.5 `/predict` — What if

Click a forest cell on the map, then move weather sliders (clamped to training p1–p99). VPD and dewpoint are derived from T+RH. Response: baseline vs scenario calibrated chance, class, grouped SHAP. Not a yes/no fire.

```
GET /api/whatif/schema
POST /api/whatif   { lat, lon, date, horizon, overrides }
```

---

## 3. How to use the APIs (client rules)

### 3.1 Dev proxy (do this first)

`frontend/vite.config.js`:

```js
server: {
  proxy: { "/api": "http://127.0.0.1:8000" },
}
```

Run API + UI together:

```bash
make api                          # :8000
cd frontend && npm run dev        # :5173, /api proxied
```

No CORS needed in dev. If the UI is ever served from another origin, add FastAPI `CORSMiddleware` then — not before.

### 3.2 Fetch layer

`src/api/client.js` — one `apiGet(path, params)` that:

- builds query strings
- throws a typed `ApiError` on 404/400 with `detail`
- returns JSON (or blob only if we ever need it; tiles go through Leaflet, not `fetch`)

Thin wrappers:

| Helper | Endpoint |
|---|---|
| `getDistricts({ date, horizon })` | `/api/districts` |
| `getDistrictTimeseries({ id, horizon, start, end })` | `/api/districts/{id}/timeseries` |
| `getActiveFires({ asOf, lookbackDays, limit })` | `/api/fires/active` |
| `getVerification({ start, end })` | `/api/verification` |
| `getExplain({ lat, lon, date, horizon, top })` | `/api/explain` |
| `getWhatIfSchema()` | `/api/whatif/schema` |
| `postWhatIf({ … })` | `/api/whatif` |
| `riskTileUrl({ date, horizon })` | template string for Leaflet |

Cache: React `useQuery`-style is optional; a small in-memory map keyed by URL is enough. District GeoJSON is ~77 features — fine to refetch on date change. Timeseries hits every `districts_*.geojson` on the server; call it on district pages only, not on every map pan.

### 3.3 Query-string state

Share `?date=2026-04-12&horizon=1` across `/`, `/predict`, `/district/:id`, `/fires`. Theme stays in `localStorage`, not the URL.

### 3.4 Error / empty states

| Case | UI |
|---|---|
| 404 missing COG / districts | “No forecast for this date (Jan–May 2024–2026).” |
| 404 no fires | “No detections in this window.” |
| 400 explain off-mask | “Outside the forest mask — no model score.” |
| API down | Banner: start `make api`. |

### 3.5 Backend follow-ups (only if the UI hits them)

Not frontend work unless they block the map:

1. **Out-of-bounds tiles** — `rio_tiler` raises `TileOutsideBounds` (500). Return a 1×1 transparent PNG or 204 so Leaflet does not spam errors at low zoom.
2. **`/api/fires/active` 404** when empty — prefer `200` + empty `FeatureCollection` so the explorer can show an empty state without treating it as an error.
3. **CORS** — only if proxy is not used.

---

## 4. Target folder layout

```
frontend/
  Design.md                 # token source of truth (do not duplicate hex in components)
  PLAN.md                   # this file
  index.html                # Google Fonts link + data-theme
  vite.config.js            # Tailwind plugin + /api proxy
  src/
    main.jsx
    App.jsx                 # BrowserRouter + ThemeProvider + shell
    index.css               # @import tailwind + tokens
    styles/tokens.css       # CSS variables for dark/light
    theme/ThemeProvider.jsx
    api/client.js
    lib/nepal.js            # bounds, center, season helpers
    lib/riskColors.js       # class → fill (legend + choropleth, not the COG)
    components/
      ui/Button.jsx
      ui/Card.jsx
      ui/Tabs.jsx
      chrome/AppHeader.jsx
      chrome/ThemeToggle.jsx
      map/RiskTileLayer.jsx
      map/DistrictLayer.jsx
      map/FirePointsLayer.jsx
      map/MapLegend.jsx
      map/DateScrubber.jsx
      map/HorizonToggle.jsx
      map/ExplainDrawer.jsx
      charts/DistrictTimeseries.jsx
      charts/VerificationSparkline.jsx
    pages/
      MapPage.jsx
      PredictPage.jsx
      DistrictPage.jsx
      FiresPage.jsx
      VerifyPage.jsx
```

Delete or stop importing: `src/page.jsx`, mock layers (`PatchGridLayer`, mock `HistoricalFiresLayer` data). Port any useful map-control ideas into the new `components/map/` files, restyled to tokens.

---

## 5. Build steps (serial)

Do these in order. Each step should leave the app runnable.

### Step 1 — Shell and tokens
- Add Google Fonts link from Design.md `fonts_url` to `index.html`.
- `tokens.css` + `data-theme` + `ThemeProvider` + header with working toggle.
- Map Tailwind colors to CSS variables (`@theme` in Tailwind 4).
- Empty routes rendering four placeholder pages in Arcade Night chrome.
- **Check:** toggle light/dark; fonts load; no slate/blue left on the shell.

### Step 2 — Vite proxy + API client
- Proxy `/api` → `:8000`.
- `client.js` wrappers + a tiny health check on `/docs` or `/api/verification`.
- **Check:** `npm run dev` with `make api` running; verification JSON in the network tab.

### Step 3 — National map (acceptance path)
- Leaflet, Nepal center zoom ~7, Carto basemap by theme.
- `RiskTileLayer` for `2025-04-12` h1.
- Date input (no animation yet) + H1/H7 tabs.
- Legend matching tile bins.
- **Check:** real COG tiles visible over Nepal.

### Step 4 — Districts on the map
- Fetch `/api/districts`, choropleth + click → `/district/:id`.
- Stats strip from feature properties.
- **Check:** 77 districts, names readable, Extreme count matches the GeoJSON.

### Step 5 — Date scrubber animation
- Season-aware range (Jan 1–May 31 of selected year; years 2024 and 2025).
- Play/pause: `setInterval` ~250–400ms, advance `date`, swap tile layer key.
- Pause on 404 / missing day (leap quirks) and skip.
- **Check:** play walks April 2025; tiles update; this is the Day 15 “done when”.

### Step 6 — Explain drawer
- Map click → lat/lon → `/api/explain`.
- List top features (name, value, SHAP). Show `collinear_twin` as a footnote, not a second bar.
- **Check:** forest cell returns six rows; non-forest shows the mask message.

### Step 7 — District page
- Timeseries chart + highlighted district on a small map.
- **Check:** a known `district_id` shows a Jan–May line for 2025.

### Step 8 — Fires page
- `/api/fires/active` points; lookback control; optional risk underlay.
- **Check:** 2025-04-12 (or a busy day) shows points; empty day is an empty state.

### Step 9 — Verification page
- Summary + table + sparkline. Base rate column visible. `valid: false` as em-dash.
- **Check:** numbers match `runs/forecasts/verification.csv` / `/api/verification`.

### Step 10 — Polish
- Loading skeletons (hairline cards, no spinners-as-decoration).
- Keyboard: play/pause on space when scrubber focused.
- `eslint` clean. Favicon/title `Prometheus`.
- Confirm Design.md checklist: fonts, button recipes, radii, no invented accents, lime used once per screen as the CTA/active state.

---

## 6. Colour ramps (do not mix)

| Layer | Ramp | Where defined |
|---|---|---|
| Risk **tiles** | Yellow → orange → red → purple | Server (`risk_tiles.py`) — client only documents it in the legend |
| District **class** choropleth | Same family, 5 class fills, low opacity | `src/lib/riskColors.js` |
| UI accent | Volt lime only | Design tokens |
| Live fires | Live red `#ff3b3b` only for a live indicator | Design.md status |

Never use the old green→red patch colors.

---

## 7. Out of scope for Day 15

- MapLibre / PostGIS / auth
- Live GEE “today” (cube ends at last exported season day)
- Editing forecasts
- New chart libraries (Recharts is enough)
- Inventing extra pages beyond Map / What if / District / Fires / Accuracy

---

## 8. Runbook (when implementing)

```bash
source .prometheus-venv/bin/activate
make api
# other terminal
cd frontend && npm run dev
```

Open `http://localhost:5173`. Confirm tiles:  
`http://localhost:5173/api/risk/tiles/0/0/0.png?date=2026-04-12&horizon=1`
