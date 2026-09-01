# Prometheus fire-risk model — v1

Frozen 2026-08-11T16:37:57+00:00. Nepal, Jan–May burning season, 1 km canonical grid (465 × 912, EPSG:4326).

## What it predicts

Probability that a 1 km forest cell contains at least one satellite fire
detection within the next *h* days, for h = 1 and h = 7.

## Year split

| Role | Years |
|---|---|
| Fitted | 2016–2023 |
| Isotonic calibration | 2025 |
| Reported (never fitted or calibrated on) | 2026 |

## Performance on the held-out season

| Horizon | Base rate | PR-AUC | ECE raw | ECE calibrated | Brier calibrated |
|---|---|---|---|---|---|
| h1 | 0.529% | 0.1538 | 0.1757 | 0.00018 | 0.00474 |
| h7 | 2.603% | 0.2875 | 0.1531 | 0.00281 | 0.02145 |

ECE uses equal-count bins; at a sub-1 % base rate equal-width bins put
almost every pixel in the first bin and report a flatteringly small number.

Calibration is not cosmetic here. Training uses a 1:20 negative downsample,
so the raw booster score averages ~0.18 against a true rate under 1 % — off
by a factor of thirty. Isotonic is fitted on **full-grid** predictions from
the calibration season, because a fit on the sampled table would map onto
the sampled prevalence and stay just as wrong.

## Latency

| Horizon | Trees | Median `predict(date)` |
|---|---|---|
| h1 | 202 | 0.340 s |
| h7 | 389 | 0.602 s |

Returns a (465, 912) calibrated surface. The first call of a season pays a
~14 s warm-up: rolling windows, dry-day counters, and fire-history state for
any single day depend on the whole season to date, so the season is built
once and cached (~3.4 GB resident). Every later date is a slice and one
booster pass.

Learning rate is 0.05 rather than the
0.02 that won the Day 9 search. Inference cost is linear in tree count, and
0.02 needed roughly twice as many trees to buy a PR-AUC difference smaller
than the fold-to-fold spread — it cost about 2 % relative PR-AUC and bought
a 3.3x latency reduction.

## Risk classes

Quantiles [0.0, 0.5, 0.75, 0.9, 0.95, 1.0] of the predicted distribution
over the calibration season give Low, Moderate, High, VeryHigh, Extreme.
Classes are relative — Extreme means the top 5 % of place-days, matching
operational fire-danger convention, not a fixed probability.

h1 on the held-out season:

| Class | % of grid | Observed rate | % of fires captured |
|---|---|---|---|
| Low | 52.3% | 0.039% | 3.9% |
| Moderate | 28.6% | 0.215% | 11.7% |
| High | 12.6% | 0.809% | 19.3% |
| VeryHigh | 3.4% | 2.046% | 13.3% |
| Extreme | 3.0% | 9.038% | 51.9% |

h7 on the held-out season:

| Class | % of grid | Observed rate | % of fires captured |
|---|---|---|---|
| Low | 55.6% | 0.353% | 7.5% |
| Moderate | 27.2% | 1.769% | 18.5% |
| High | 12.0% | 5.588% | 25.7% |
| VeryHigh | 2.8% | 13.509% | 14.5% |
| Extreme | 2.5% | 35.823% | 33.8% |

## Intended use and limits

- Jan–May only. Nov–Dec fires (~8 % of detections) are out of scope.
- Forest mask only (126,622 cells); no prediction is made elsewhere.
- Labels are satellite *detections*, so cloud and overpass gaps mean
  absence of a detection is not proof of absence of fire.
- Fire history carries the model (removing it halves PR-AUC), so skill
  degrades in cells with no recorded history.
- Static human and terrain layers add nothing measurable; do not read the
  model as evidence about roads or settlements.
- Raw scores are inflated by the 1:20 training downsample. Always use the
  calibrated output; the raw booster margin is not a probability.

## Reproducing

```bash
python scripts/build_model_bundle.py
python scripts/plot_calibration.py
```
