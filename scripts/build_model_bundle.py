#!/usr/bin/env python3
"""Day 11: calibrate, add the 7-day horizon, freeze a versioned bundle.

  python scripts/build_model_bundle.py

Year split, chosen so nothing measured is anything the model saw:
  fit 2016-2023 · early stopping 2024 · isotonic calibration 2025 · report 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prometheus.config import load_settings
from prometheus.features import derived
from prometheus.features import table as ftable
from prometheus.models import calibrate as cal
from prometheus.models import lgbm
from prometheus.models.bundle import HorizonArtifacts, ModelBundle, next_version


def grid_truth(bundle: dict, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Flatten labels and the valid-day mask for a scored season."""
    labels = derived.horizon_labels(bundle["fire"], [horizon])
    usable = labels[f"valid_h{horizon}"]
    y = labels[f"label_h{horizon}"][:, bundle["rows"], bundle["cols"]][usable]
    return y.ravel(), usable


def class_table(y_true, probabilities, thresholds, names) -> pd.DataFrame:
    """Share of the grid and observed fire rate in each risk class."""
    idx = cal.classify(probabilities, thresholds)
    rows = []
    for k, name in enumerate(names):
        sel = idx == k
        rows.append(
            {
                "class": name,
                "share_of_grid": float(sel.mean()),
                "observed_rate": float(y_true[sel].mean()) if sel.any() else float("nan"),
                "fires_captured": float(y_true[sel].sum() / max(y_true.sum(), 1)),
                "mean_probability": (
                    float(np.mean(probabilities[sel])) if sel.any() else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_horizon(
    table, horizon: int, *, calib_year: int, test_year: int, cube, fire_ds, anomaly,
    settings, out_dir: Path, learning_rate: float,
) -> tuple[HorizonArtifacts, dict]:
    label = f"label_h{horizon}"
    print(f"\n{'=' * 72}\nHORIZON h{horizon} — {label}\n{'=' * 72}")

    # Holding out the calibration year makes early stopping fall on the year
    # before it, so neither the calibration nor the test season is ever fitted.
    fit_table = table[table["year"] <= calib_year]
    booster, result = lgbm.train_fold(
        fit_table, calib_year, label=label, params={"learning_rate": learning_rate}
    )
    features = list(booster.feature_name())
    print(f"  fit on {sorted(set(fit_table['year']) - {calib_year, result.year})}"
          f" · {result.best_iteration} trees · {result.train_seconds:.0f}s")

    scored = {}
    for tag, year in (("calibration", calib_year), ("test", test_year)):
        print(f"  scoring {tag} season {year} on the full grid …", flush=True)
        grid_bundle = ftable.year_grid_features(
            year, cube=cube, fire_ds=fire_ds, anomaly=anomaly, features=features
        )
        y, usable = grid_truth(grid_bundle, horizon)
        raw = lgbm.score_matrix(booster, grid_bundle)[usable].ravel()
        scored[tag] = {"year": year, "y": y, "raw": raw}
        del grid_bundle

    calibrator = cal.fit_isotonic(
        scored["calibration"]["y"], scored["calibration"]["raw"], fit_year=calib_year
    )
    print(f"  isotonic fit on {calibrator.n_fit:,} pixel-days "
          f"(base rate {calibrator.base_rate:.4%}, {calibrator.x.size} breakpoints)")

    calib_prob = calibrator(scored["calibration"]["raw"])
    thresholds = cal.quantile_thresholds(calib_prob, settings.risk_classes.quantiles)
    print("  risk thresholds: " + ", ".join(f"{t:.5f}" for t in thresholds))

    test = scored["test"]
    test_prob = calibrator(test["raw"])
    report = cal.calibration_report(test["y"], test["raw"], test_prob)
    report["horizon"] = horizon
    report["test_year"] = test_year
    report["calibration_year"] = calib_year

    print(f"\n  reliability on held-out {test_year} "
          f"(base rate {report['base_rate']:.4%}, {report['n']:,} pixel-days)")
    print(f"    mean predicted   raw {report['mean_raw']:.4f} → "
          f"calibrated {report['mean_calibrated']:.5f}")
    print(f"    ECE (equal-count) raw {report['ece_quantile_raw']:.4f} → "
          f"calibrated {report['ece_quantile_calibrated']:.5f}")
    print(f"    ECE (equal-width) raw {report['ece_uniform_raw']:.4f} → "
          f"calibrated {report['ece_uniform_calibrated']:.5f}")
    print(f"    Brier             raw {report['brier_raw']:.4f} → "
          f"calibrated {report['brier_calibrated']:.5f}")
    # Isotonic is monotone, so ranking survives except where the fit maps
    # distinct raw scores to one value; those ties cost a little PR-AUC.
    print(f"    PR-AUC            raw {report['pr_auc_raw']:.4f} → "
          f"calibrated {report['pr_auc_calibrated']:.4f} "
          f"({report['pr_auc_calibrated'] - report['pr_auc_raw']:+.4f} from isotonic ties)")

    names = settings.risk_classes.names
    classes = class_table(test["y"], test_prob, thresholds, names)
    print(f"\n  risk classes on {test_year}:")
    print(f"    {'class':<12}{'% of grid':>11}{'obs. rate':>11}"
          f"{'% of fires':>12}{'mean prob':>11}")
    for _, r in classes.iterrows():
        print(f"    {r['class']:<12}{r.share_of_grid:>10.1%}{r.observed_rate:>11.4%}"
              f"{r.fires_captured:>11.1%}{r.mean_probability:>11.5f}")

    np.savez_compressed(
        out_dir / f"reliability_h{horizon}.npz",
        y=test["y"].astype(np.uint8),
        raw=test["raw"].astype(np.float32),
        calibrated=test_prob.astype(np.float32),
    )
    classes.to_csv(out_dir / f"risk_classes_h{horizon}.csv", index=False)
    report["classes"] = classes.to_dict(orient="records")

    artifacts = HorizonArtifacts(
        horizon=horizon,
        model_file=str(lgbm.save_model(booster, result, tag=f"bundle_h{horizon}")),
        features=features,
        calibrator=calibrator,
        risk_thresholds=thresholds,
        metrics={
            **{k: v for k, v in report.items() if k != "classes"},
            "n_trees": int(booster.num_trees()),
            "learning_rate": learning_rate,
        },
    )
    return artifacts, report


def model_card(bundle: ModelBundle, reports: dict[int, dict], settings) -> str:
    lines = [
        f"# Prometheus fire-risk model — {bundle.version}",
        "",
        f"Frozen {bundle.created_at}. Nepal, Jan–May burning season, "
        "1 km canonical grid (465 × 912, EPSG:4326).",
        "",
        "## What it predicts",
        "",
        "Probability that a 1 km forest cell contains at least one satellite fire",
        "detection within the next *h* days, for h = 1 and h = 7.",
        "",
        "## Year split",
        "",
        "| Role | Years |",
        "|---|---|",
        f"| Fitted | {bundle.train_years[0]}–{bundle.train_years[-1]} |",
        f"| Isotonic calibration | {bundle.calibration_year} |",
        f"| Reported (never fitted or calibrated on) | {bundle.test_year} |",
        "",
        "## Performance on the held-out season",
        "",
        "| Horizon | Base rate | PR-AUC | ECE raw | ECE calibrated | Brier calibrated |",
        "|---|---|---|---|---|---|",
    ]
    for h, r in sorted(reports.items()):
        lines.append(
            f"| h{h} | {r['base_rate']:.3%} | {r['pr_auc_calibrated']:.4f} | "
            f"{r['ece_quantile_raw']:.4f} | {r['ece_quantile_calibrated']:.5f} | "
            f"{r['brier_calibrated']:.5f} |"
        )
    lines += [
        "",
        "ECE uses equal-count bins; at a sub-1 % base rate equal-width bins put",
        "almost every pixel in the first bin and report a flatteringly small number.",
        "",
        "Calibration is not cosmetic here. Training uses a 1:20 negative downsample,",
        "so the raw booster score averages ~0.18 against a true rate under 1 % — off",
        "by a factor of thirty. Isotonic is fitted on **full-grid** predictions from",
        "the calibration season, because a fit on the sampled table would map onto",
        "the sampled prevalence and stay just as wrong.",
        "",
        "## Latency",
        "",
        "| Horizon | Trees | Median `predict(date)` |",
        "|---|---|---|",
    ]
    for h, artifacts in sorted(bundle.horizons.items()):
        trees = artifacts.metrics.get("n_trees", 0)
        latency = bundle.notes.get("latency_seconds", {}).get(str(h))
        shown = f"{latency:.3f} s" if latency else "—"
        lines.append(f"| h{h} | {trees} | {shown} |")
    lines += [
        "",
        "Returns a (465, 912) calibrated surface. The first call of a season pays a",
        "~14 s warm-up: rolling windows, dry-day counters, and fire-history state for",
        "any single day depend on the whole season to date, so the season is built",
        "once and cached (~3.4 GB resident). Every later date is a slice and one",
        "booster pass.",
        "",
        f"Learning rate is {bundle.notes.get('learning_rate', 0.05)} rather than the",
        "0.02 that won the Day 9 search. Inference cost is linear in tree count, and",
        "0.02 needed roughly twice as many trees to buy a PR-AUC difference smaller",
        "than the fold-to-fold spread — it cost about 2 % relative PR-AUC and bought",
        "a 3.3x latency reduction.",
        "",
        "## Risk classes",
        "",
        f"Quantiles {settings.risk_classes.quantiles} of the predicted distribution",
        f"over the calibration season give {', '.join(settings.risk_classes.names)}.",
        "Classes are relative — Extreme means the top 5 % of place-days, matching",
        "operational fire-danger convention, not a fixed probability.",
        "",
    ]
    for h, r in sorted(reports.items()):
        classes = r.get("classes", [])
        if not classes:
            continue
        lines += [
            f"h{h} on the held-out season:",
            "",
            "| Class | % of grid | Observed rate | % of fires captured |",
            "|---|---|---|---|",
        ]
        for row in classes:
            lines.append(
                f"| {row['class']} | {row['share_of_grid']:.1%} | "
                f"{row['observed_rate']:.3%} | {row['fires_captured']:.1%} |"
            )
        lines.append("")
    lines += [
        "## Intended use and limits",
        "",
        "- Jan–May only. Nov–Dec fires (~8 % of detections) are out of scope.",
        "- Forest mask only (126,622 cells); no prediction is made elsewhere.",
        "- Labels are satellite *detections*, so cloud and overpass gaps mean",
        "  absence of a detection is not proof of absence of fire.",
        "- Fire history carries the model (removing it halves PR-AUC), so skill",
        "  degrades in cells with no recorded history.",
        "- Static human and terrain layers add nothing measurable; do not read the",
        "  model as evidence about roads or settlements.",
        "- Raw scores are inflated by the 1:20 training downsample. Always use the",
        "  calibrated output; the raw booster margin is not a probability.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python scripts/build_model_bundle.py",
        "python scripts/plot_calibration.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    years = sorted(settings.years.all)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-year", type=int, default=years[-2])
    ap.add_argument("--test-year", type=int, default=years[-1])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 7])
    # Inference cost is linear in tree count (~1.6 ms per tree over the mask), and
    # the Day 9 search found learning rate to be worth less than fold noise. 0.05
    # buys the sub-second latency budget for no measurable accuracy.
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--version", default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or (Path(settings.root) / "runs" / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    table = ftable.load_train_table()
    cube = ftable.open_cube()
    fire_ds = ftable._fire_cube()
    anomaly = ftable.build_anomaly(cube, years)

    print(f"{len(table):,} rows · calibrate on {args.calib_year} · report on {args.test_year}")

    horizons, reports = {}, {}
    for horizon in args.horizons:
        artifacts, report = build_horizon(
            table, horizon, calib_year=args.calib_year, test_year=args.test_year,
            cube=cube, fire_ds=fire_ds, anomaly=anomaly, settings=settings,
            out_dir=out_dir, learning_rate=args.learning_rate,
        )
        horizons[horizon] = artifacts
        reports[horizon] = report

    fitted = [y for y in years if y < args.calib_year - 1]
    bundle = ModelBundle(
        version=args.version or next_version(),
        horizons=horizons,
        train_years=fitted,
        calibration_year=args.calib_year,
        test_year=args.test_year,
        risk_class_names=list(settings.risk_classes.names),
        risk_quantiles=list(settings.risk_classes.quantiles),
        notes={
            "early_stopping_year": args.calib_year - 1,
            "learning_rate": args.learning_rate,
        },
    )
    root = bundle.save()
    (out_dir / "calibration_report.json").write_text(
        json.dumps({str(h): r for h, r in reports.items()}, indent=2, default=float),
        encoding="utf-8",
    )

    print(f"\n{'=' * 72}\nfrozen bundle {bundle.version} → {root}")

    ok, latency = benchmark(bundle, args.test_year)
    bundle.notes["latency_seconds"] = latency
    bundle.save(root)
    (root / "MODEL_CARD.md").write_text(
        model_card(bundle, reports, settings), encoding="utf-8"
    )
    print(f"\nmodel card → {root / 'MODEL_CARD.md'}")
    for name in sorted(p.name for p in root.iterdir()):
        print(f"  {name}")
    return 0 if ok else 1


def benchmark(
    bundle: ModelBundle, year: int, *, budget: float = 1.0
) -> tuple[bool, dict[str, float]]:
    """Time `predict(date)` on the frozen bundle — the Day 11 acceptance gate."""
    import time

    from prometheus.models.predict import RiskPredictor

    print(f"\n{'=' * 72}\nLATENCY — predict(date) → (465, 912)\n{'=' * 72}")
    predictor = RiskPredictor(bundle)

    start = time.perf_counter()
    predictor.warm(year)
    print(f"  season warm-up (once per season): {time.perf_counter() - start:.2f}s")

    dates = predictor._year_features(year)["dates"]
    probe = [dates[i] for i in np.linspace(0, len(dates) - 1, 7).astype(int)]
    ok = True
    latency: dict[str, float] = {}
    for horizon in predictor.horizons:
        timings = []
        for day in probe:
            start = time.perf_counter()
            surface = predictor.predict(day, horizon=horizon)
            timings.append(time.perf_counter() - start)
        trees = bundle.horizons[horizon].metrics.get("n_trees", 0)
        median = float(np.median(timings))
        latency[str(horizon)] = median
        passed = max(timings) < budget
        ok &= passed
        print(f"  h{horizon}: {trees} trees · median {median:.3f}s · "
              f"max {max(timings):.3f}s · shape {surface.shape} · "
              f"{'PASS' if passed else 'FAIL'} (<{budget:.0f}s)")
    return ok, latency


if __name__ == "__main__":
    raise SystemExit(main())
