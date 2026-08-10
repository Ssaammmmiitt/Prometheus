#!/usr/bin/env python3
"""Day 6-7 verification plot: one Terai forest cell through a fire season.

  python scripts/plot_cube_check.py --year 2021
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from prometheus import grid  # noqa: E402
from prometheus.config import load_settings  # noqa: E402
from prometheus.features import forest  # noqa: E402
from prometheus.features.cube import open_cube  # noqa: E402

TERAI_CODE = 1


def pick_terai_cell(ds) -> tuple[int, int]:
    """Median-elevation Terai forest cell, so the plot is representative."""
    mask = grid.nepal_mask() & forest.forest_mask()
    if "physio_region" in ds:
        mask &= ds["physio_region"].values == TERAI_CODE
    elev = ds["elevation"].values
    cells = np.argwhere(mask)
    if not len(cells):
        raise RuntimeError("No Terai forest cells found")
    heights = elev[mask]
    order = np.argsort(heights)
    return tuple(int(v) for v in cells[order[len(order) // 2]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = open_cube()
    row, col = pick_terai_cell(ds)
    elev = float(ds["elevation"].values[row, col])
    lat = float(ds["y"].values[row])
    lon = float(ds["x"].values[col])

    sub = ds.isel(y=row, x=col).sel(time=str(args.year))
    times = sub["time"].values

    def series(name: str) -> np.ndarray:
        return sub[name].values.astype(np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        f"Prometheus feature cube — Terai forest cell "
        f"({lat:.3f}N, {lon:.3f}E, {elev:.0f} m), {args.year} fire season",
        fontsize=12,
    )

    ax = axes[0]
    ax.plot(times, series("t2m_max"), color="#c0392b", lw=1.2, label="T max")
    ax.plot(times, series("t2m"), color="#e67e22", lw=1.0, label="T mean")
    ax.plot(times, series("t2m_min"), color="#2980b9", lw=1.0, label="T min")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(times, series("rh"), color="#16a085", lw=1.2, label="RH")
    ax.set_ylabel("Relative humidity (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(times, series("vpd"), color="#8e44ad", lw=1.0, ls="--", label="VPD")
    twin.set_ylabel("VPD (kPa)")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(times, series("wind_speed"), color="#2c3e50", lw=1.0, label="wind speed")
    ax.set_ylabel("Wind (m s$^{-1}$)")
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.bar(times, series("precip"), color="#3498db", alpha=0.5, width=1.0, label="precip")
    twin.set_ylabel("Precip (mm)")
    lines = ax.get_lines() + [twin.containers[0]]
    ax.legend(lines, ["wind speed", "precip"], loc="upper left", fontsize=8)

    ax = axes[3]
    ax.plot(times, series("ndvi"), color="#27ae60", lw=1.2, label="NDVI")
    ax.set_ylabel("NDVI")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(times, series("lst_day"), color="#d35400", lw=1.0, ls="--", label="LST day")
    twin.set_ylabel("LST day (°C)")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower left", fontsize=8)
    ax.set_xlabel("Date")

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or (Path(load_settings().root) / "runs" / "cube" / f"terai_cell_{args.year}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"saved {out}")

    print(f"\nCell: row={row} col={col} lat={lat:.4f} lon={lon:.4f} elev={elev:.0f} m")
    print(f"{'variable':<14}{'Jan':>9}{'Mar':>9}{'Apr':>9}{'May':>9}")
    months = {"Jan": 1, "Mar": 3, "Apr": 4, "May": 5}
    mon = sub["time"].dt.month.values
    for name in ("t2m_max", "rh", "vpd", "wind_speed", "precip", "ndvi", "lst_day"):
        vals = series(name)
        row_txt = f"{name:<14}"
        for m in months.values():
            row_txt += f"{np.nanmean(vals[mon == m]):9.2f}"
        print(row_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
