"""ERA5-Land daily weather: 9 km → 1 km with lapse-rate corrected temperature."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

from prometheus.config import load_settings
from prometheus.features.warp import coarse_elevation, elevation_1km, warp_array

# Environmental lapse rate (K m-1). Standard atmosphere value used across the
# downscaling literature; Nepal's relief spans >8 km so this term is large.
LAPSE_T = 0.0065
# Dewpoint falls more slowly than air temperature with height (~2 K km-1).
# Using the dry-air rate for both would leave relative humidity unchanged and
# throw away the very signal we are downscaling for.
LAPSE_TD = 0.0020

G = 9.80665  # m s-2
R_DRY = 287.05  # J kg-1 K-1

RAW_VARS = (
    "t2m_max",
    "t2m_min",
    "t2m",
    "d2m",
    "precip",
    "u10",
    "v10",
    "soil_water_l1",
    "surface_pressure",
)
DERIVED_VARS = ("rh", "vpd", "wind_speed")
WEATHER_VARS = RAW_VARS + DERIVED_VARS

_BAND_RE = re.compile(r"^(?:(\d{8})_)?(\d{8})_(.+)$")


def era5_dir() -> Path:
    return load_settings().paths.resolve("gee_raw") / "era5"


def month_path(year: int, month: int) -> Path:
    return era5_dir() / f"era5_{year}_{month:02d}.tif"


def parse_band(name: str) -> tuple[date, str] | None:
    """'20210401_20210401_t2m_max' -> (date(2021,4,1), 't2m_max')."""
    m = _BAND_RE.match(name or "")
    if not m:
        return None
    stamp = m.group(2)
    var = m.group(3)
    try:
        d = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
    except ValueError:
        return None
    return d, var


def read_month(year: int, month: int) -> tuple[list[date], dict[str, np.ndarray], object]:
    """Read one monthly ERA5 stack at native ~9 km.

    Returns (dates, {var: (n_days, h, w)}, src_transform).
    """
    path = month_path(year, month)
    if not path.is_file():
        raise FileNotFoundError(f"ERA5 month missing: {path}")

    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)
        transform = src.transform
        index: dict[str, dict[date, int]] = {}
        for i, desc in enumerate(descriptions, start=1):
            parsed = parse_band(desc)
            if parsed is None:
                continue
            d, var = parsed
            index.setdefault(var, {})[d] = i

        if not index:
            raise ValueError(f"No parseable ERA5 band names in {path.name}")

        dates = sorted(set.intersection(*(set(v) for v in index.values())))
        out: dict[str, np.ndarray] = {}
        for var, per_date in index.items():
            bands = [per_date[d] for d in dates]
            out[var] = src.read(bands).astype(np.float32)

    return dates, out, transform


def saturation_vapour_pressure(t_c: np.ndarray) -> np.ndarray:
    """Magnus-Tetens saturation vapour pressure over water, hPa."""
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def relative_humidity(t2m_c: np.ndarray, d2m_c: np.ndarray) -> np.ndarray:
    rh = 100.0 * saturation_vapour_pressure(d2m_c) / saturation_vapour_pressure(t2m_c)
    return np.clip(rh, 1.0, 100.0)


def vapour_pressure_deficit(t2m_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """VPD in kPa (hPa / 10)."""
    es = saturation_vapour_pressure(t2m_c) / 10.0
    return np.maximum(es * (1.0 - rh_pct / 100.0), 0.0)


def dewpoint_from_rh(t2m_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Invert Magnus–Tetens: dewpoint (°C) from air temperature and RH."""
    t = np.asarray(t2m_c, dtype=np.float64)
    rh = np.clip(np.asarray(rh_pct, dtype=np.float64), 1.0, 100.0)
    vapour = saturation_vapour_pressure(t) * (rh / 100.0)
    vapour = np.maximum(vapour, 1e-6)
    ln = np.log(vapour / 6.112)
    d2m = 243.5 * ln / (17.67 - ln)
    return np.minimum(d2m, t).astype(np.float32)


def downscale_month(year: int, month: int) -> tuple[list[date], dict[str, np.ndarray]]:
    """
    ERA5 month at ~9 km → daily 1 km fields with derived humidity and wind.

    Temperature is lapse-rate corrected against the sub-grid terrain that the
    ~9 km grid averages away:
        T_1km = T_era5 + LAPSE * (elev_era5 - elev_1km)
    """
    dates, coarse, src_transform = read_month(year, month)

    elev_fine = elevation_1km()
    elev_coarse = coarse_elevation(
        tuple(float(v) for v in src_transform[:6]),
        tuple(int(v) for v in coarse["t2m"].shape[1:]),
    )
    delta_z = elev_coarse - elev_fine  # positive where the 1 km cell sits lower

    fine: dict[str, np.ndarray] = {}
    for var in RAW_VARS:
        if var not in coarse:
            continue
        fine[var] = warp_array(
            coarse[var], src_transform, resampling="bilinear"
        ).astype(np.float32)

    for var in ("t2m", "t2m_max", "t2m_min"):
        if var in fine:
            fine[var] = fine[var] + LAPSE_T * delta_z
    if "d2m" in fine:
        fine["d2m"] = fine["d2m"] + LAPSE_TD * delta_z
        # Dewpoint cannot exceed air temperature.
        if "t2m" in fine:
            fine["d2m"] = np.minimum(fine["d2m"], fine["t2m"])

    if "surface_pressure" in fine:
        # Pa → hPa, then hypsometric adjustment to the 1 km surface. Keeping Pa
        # would overflow the float16 cube (max 65504).
        t_kelvin = fine.get("t2m", np.zeros_like(delta_z)) + 273.15
        t_kelvin = np.where(np.isfinite(t_kelvin) & (t_kelvin > 150), t_kelvin, 273.15)
        fine["surface_pressure"] = (
            fine["surface_pressure"] / 100.0 * np.exp(G * delta_z / (R_DRY * t_kelvin))
        ).astype(np.float32)

    if "precip" in fine:
        fine["precip"] = np.maximum(fine["precip"], 0.0)
    if "soil_water_l1" in fine:
        fine["soil_water_l1"] = np.clip(fine["soil_water_l1"], 0.0, 1.0)

    if "t2m" in fine and "d2m" in fine:
        fine["rh"] = relative_humidity(fine["t2m"], fine["d2m"]).astype(np.float32)
        fine["vpd"] = vapour_pressure_deficit(fine["t2m"], fine["rh"]).astype(np.float32)
    if "u10" in fine and "v10" in fine:
        fine["wind_speed"] = np.hypot(fine["u10"], fine["v10"]).astype(np.float32)

    return dates, fine


__all__ = [
    "DERIVED_VARS",
    "LAPSE_T",
    "LAPSE_TD",
    "RAW_VARS",
    "WEATHER_VARS",
    "downscale_month",
    "era5_dir",
    "month_path",
    "parse_band",
    "read_month",
    "dewpoint_from_rh",
    "relative_humidity",
    "saturation_vapour_pressure",
    "vapour_pressure_deficit",
]
