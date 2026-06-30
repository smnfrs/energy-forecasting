"""Physics helper functions for weather feature engineering.

Pure functions — each takes pandas Series and returns a Series.
Ported from EMA's data_modules/feature_eng.py (lines 74-213).
"""

import numpy as np
import pandas as pd

# Constants
R_DRY = 287.05  # J/(kg·K) — specific gas constant for dry air
R_VAPOR = 461.50  # J/(kg·K) — specific gas constant for water vapor
EARTH_RADIUS_KM = 6371.0


def air_density_dry(pressure_hpa: pd.Series, temperature_c: pd.Series) -> pd.Series:
    """Dry air density: ρ = P / (R_d * T)."""
    return (pressure_hpa * 100.0) / (R_DRY * (temperature_c + 273.15))


def air_density_moist(
    temperature_c: pd.Series,
    pressure_hpa: pd.Series,
    humidity_pct: pd.Series,
) -> pd.Series:
    """Moist air density using partial pressures of dry air and water vapor."""
    t_k = temperature_c + 273.15
    p_total_pa = pressure_hpa * 100.0
    e_pa = vapor_pressure(temperature_c, humidity_pct) * 100.0
    p_dry_pa = p_total_pa - e_pa
    return (p_dry_pa / (R_DRY * t_k)) + (e_pa / (R_VAPOR * t_k))


def wind_power_density(wind_speed_kmh: pd.Series, density_kg_m3: pd.Series) -> pd.Series:
    """Wind power density: P/A = 0.5 * ρ * v³ [W/m²]. Input wind in km/h."""
    v_ms = wind_speed_kmh / 3.6
    return 0.5 * density_kg_m3 * v_ms**3


def dew_point_temperature(temperature_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """Dew point via Magnus formula."""
    a, b = 17.62, 243.12
    rh_safe = humidity_pct.clip(lower=0.1) / 100.0
    gamma = np.log(rh_safe) + (a * temperature_c / (b + temperature_c))
    return (b * gamma) / (a - gamma)


def vapor_pressure(temperature_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """Saturation vapor pressure × RH [hPa]."""
    rh_frac = humidity_pct / 100.0
    return 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5)) * rh_frac


def wind_shear(wind_high: pd.Series, wind_low: pd.Series) -> pd.Series:
    """Log-ratio wind shear between two heights.

    When ``wind_low`` is zero (calm conditions) the ratio is undefined; we
    return 0 (no detectable shear) to keep the column NaN-free. NaN is
    avoided so per-farm features remain alignable across different weather
    sources (e.g. history vs hist_forecast may have different zero-wind
    hours, causing reindex-NaN holes if formulas leave NaN in the output).
    """
    ratio = wind_high / wind_low.replace({0: np.nan})
    shear = np.log(np.maximum(ratio, 1e-10)) / np.log(10.0)
    return shear.replace([np.inf, -np.inf, np.nan], 0.0)


def turbulence_intensity(wind_speed: pd.Series, window: int) -> pd.Series:
    """TI = rolling_std / rolling_mean. Returns 0 where mean is zero
    (calm conditions, undefined TI) — see ``wind_shear`` for rationale.
    """
    std = wind_speed.rolling(window=window).std()
    mean = wind_speed.rolling(window=window).mean()
    return (std / mean).replace([np.inf, -np.inf, np.nan], 0.0)


def wind_ramp(wind_speed: pd.Series) -> pd.Series:
    """First-order difference of wind speed.

    The first row's diff is NaN by construction; a single boundary NaN is
    fine for our pipeline (`_build_features` drops it via the all-NaN-row
    filter), unlike scattered mid-series NaN from undefined-ratio formulas.
    """
    return wind_speed.diff()


def gust_factor(wind_speed_kmh: pd.Series, wind_gust_kmh: pd.Series) -> pd.Series:
    """Gust factor = gust / mean wind speed (both in m/s).

    Returns 0 where wind_speed is zero (calm — undefined ratio) — see
    ``wind_shear`` for rationale.
    """
    gust_ms = wind_gust_kmh / 3.6
    ws_ms = (wind_speed_kmh / 3.6).replace({0: np.nan})
    return (gust_ms / ws_ms).replace([np.inf, -np.inf, np.nan], 0.0)


def wind_chill(temperature_c: pd.Series, wind_speed_kmh: pd.Series) -> pd.Series:
    """Wind chill index."""
    v_ms = wind_speed_kmh / 3.6
    return (
        13.12 + 0.6215 * temperature_c - 11.37 * v_ms**0.16 + 0.3965 * temperature_c * v_ms**0.16
    )


def humidex(temperature_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """Humidex (perceived temperature including humidity)."""
    e_pa = vapor_pressure(temperature_c, humidity_pct) * 100.0
    return temperature_c + 0.5555 * (e_pa / 100.0 - 10.0)


def heating_degree_hours(temperature_c: pd.Series, threshold: float = 18.0) -> pd.Series:
    """HDH = max(threshold - T, 0)."""
    return (threshold - temperature_c).clip(lower=0)


def cooling_degree_hours(temperature_c: pd.Series, threshold: float = 22.0) -> pd.Series:
    """CDH = max(T - threshold, 0)."""
    return (temperature_c - threshold).clip(lower=0)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
