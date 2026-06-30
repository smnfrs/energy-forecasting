"""Wind power weather feature engineering.

Transforms raw weather variables (wind speed, pressure, temperature, etc.)
into features relevant for wind power generation forecasting.

Ported from EMA's WeatherWindPowerFE class.
Each instance handles one asset type (onshore or offshore) for one TSO.
"""

import numpy as np
import pandas as pd

from energy_forecasting.features.spatial import aggregate_locations
from energy_forecasting.features.weather_physics import (
    air_density_dry,
    air_density_moist,
    dew_point_temperature,
    gust_factor,
    turbulence_intensity,
    vapor_pressure,
    wind_power_density,
    wind_ramp,
    wind_shear,
)

# Default config — all features enabled, moderate lags
DEFAULT_CONFIG: dict = {
    "compute_air_density": True,
    "compute_air_density_moist": False,
    "encode_wind_direction": True,
    "compute_wind_shear": True,
    "compute_wind_ramp": True,
    "gust_factor": True,
    "dew_point_temperature": False,
    "vapor_pressure": False,
    "lags_choice": "small",
    "precip_lags_choice": "none",
    "turbulence_window": 6,
    "drop_raw_main_features": False,
    "drop_raw_wind_features": False,
    "spatial_agg_method": "capacity",
}

# Raw "main" features that EMA's `drop_raw_main_features` toggles. EMA's
# `drop_raw_wind_features` also drops these (its `wind_features` list is
# misnamed — it contains the same main meteo columns). We replicate that
# behaviour: either flag set to True drops the same 4 columns.
_RAW_MAIN_FEATURES = [
    "temperature_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
]
# Raw wind columns kept regardless of drop flags (wind_speed_100m is the
# core input for derived wind features).
_RAW_WIND_FEATURES = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_gusts_10m",
    "relative_humidity_2m",
]

_LAGS = {"none": [], "small": [1, 6], "large": [1, 6, 12]}
_PRECIP_LAGS = {"none": [], "small": [1, 6], "large": [1, 6, 12, 24]}


class WeatherWindPowerFE:
    """Wind power weather feature engineering for a single TSO/asset type."""

    def __init__(self, config: dict, locations: list[dict]):
        self.config = {**DEFAULT_CONFIG, **config}
        self.locations = locations
        self.suffixes = [loc["suffix"] for loc in locations]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all locations, then spatially aggregate."""
        frames = []
        for loc in self.locations:
            loc_df = self._process_location(df, loc["suffix"])
            frames.append(loc_df)

        combined = pd.concat(frames, axis=1)

        method = self.config["spatial_agg_method"]
        if method != "None" and len(self.locations) > 1:
            combined = aggregate_locations(combined, self.suffixes, method, self.locations)

        return combined

    def _process_location(self, df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """Compute wind features for a single location."""
        result = pd.DataFrame(index=df.index)
        cfg = self.config

        # Helper to get column with suffix
        def col(name: str) -> str:
            return f"{name}{suffix}"

        def get(name: str) -> pd.Series:
            return df[col(name)]

        # Wind speed columns
        ws100 = get("wind_speed_100m") if col("wind_speed_100m") in df.columns else None
        ws10 = get("wind_speed_10m") if col("wind_speed_10m") in df.columns else None
        temp = get("temperature_2m") if col("temperature_2m") in df.columns else None
        pressure = get("surface_pressure") if col("surface_pressure") in df.columns else None
        humidity = (
            get("relative_humidity_2m") if col("relative_humidity_2m") in df.columns else None
        )

        # Air density + wind power density. Match EMA: both are tied to
        # `compute_air_density` — when False, neither is emitted (EMA's
        # `compute_wind_power_density` config flag is dead code in their
        # `_preprocess_location`, so we don't surface it as a separate
        # search variable).
        if cfg["compute_air_density"] and pressure is not None and temp is not None:
            density = air_density_dry(pressure, temp)
            result[f"air_density{suffix}"] = density
            if ws100 is not None:
                result[f"wind_power_density{suffix}"] = wind_power_density(ws100, density)

        if cfg["compute_air_density_moist"] and humidity is not None:
            result[f"air_density_moist{suffix}"] = air_density_moist(temp, pressure, humidity)

        # Wind direction encoding
        if cfg["encode_wind_direction"]:
            wd_col = col("wind_direction_100m")
            if wd_col in df.columns:
                wd_rad = np.deg2rad(df[wd_col])
                result[f"wind_dir_sin{suffix}"] = np.sin(wd_rad)
                result[f"wind_dir_cos{suffix}"] = np.cos(wd_rad)

        # Wind shear (100m vs 10m)
        if cfg["compute_wind_shear"] and ws100 is not None and ws10 is not None:
            result[f"wind_shear{suffix}"] = wind_shear(ws100, ws10)

        # Turbulence intensity
        window = cfg["turbulence_window"]
        if window > 0 and ws100 is not None:
            result[f"turbulence_intensity{suffix}"] = turbulence_intensity(ws100, window)

        # Wind ramp
        if cfg["compute_wind_ramp"] and ws100 is not None:
            result[f"wind_ramp{suffix}"] = wind_ramp(ws100)

        # Gust factor
        if cfg["gust_factor"]:
            gust_col = col("wind_gusts_10m")
            if gust_col in df.columns and ws10 is not None:
                result[f"gust_factor{suffix}"] = gust_factor(ws10, df[gust_col])

        # Dew point temperature
        if cfg["dew_point_temperature"] and temp is not None and humidity is not None:
            result[f"dew_point{suffix}"] = dew_point_temperature(temp, humidity)

        # Vapor pressure
        if cfg["vapor_pressure"] and temp is not None and humidity is not None:
            result[f"vapor_pressure{suffix}"] = vapor_pressure(temp, humidity)

        # Lags
        lags = _LAGS.get(cfg["lags_choice"], [])
        if ws100 is not None:
            for lag in lags:
                result[f"wind_speed_100m_lag{lag}{suffix}"] = ws100.shift(lag)

        precip_lags = _PRECIP_LAGS.get(cfg["precip_lags_choice"], [])
        precip_col = col("precipitation")
        if precip_col in df.columns:
            precip = df[precip_col]
            for lag in precip_lags:
                result[f"precipitation_lag{lag}{suffix}"] = precip.shift(lag)

        # Raw features. EMA starts from all suffix-matching columns and
        # conditionally drops; we start empty and conditionally include.
        # Net behaviour matches: include the four "main" meteo columns
        # unless either drop flag is True (EMA's two flags drop the same
        # list — see _RAW_MAIN_FEATURES comment). Wind columns + humidity
        # are always retained.
        drop_main = cfg["drop_raw_main_features"] or cfg["drop_raw_wind_features"]
        if not drop_main:
            for raw in _RAW_MAIN_FEATURES:
                rc = col(raw)
                if rc in df.columns:
                    result[rc] = df[rc]

        for raw in _RAW_WIND_FEATURES:
            rc = col(raw)
            if rc in df.columns:
                result[rc] = df[rc]

        # Wind direction is dropped if encode_wind_direction=True; otherwise retained.
        wd_col = col("wind_direction_100m")
        if not cfg["encode_wind_direction"] and wd_col in df.columns:
            result[wd_col] = df[wd_col]

        return result

    @staticmethod
    def suggest_optuna(trial, prefix: str = "wind") -> dict:
        """Suggest hyperparameters for Optuna tuning.

        Search space matches EMA's WeatherWindPowerFE.selector_for_optuna:
        - 11 boolean toggles (incl. vapor_pressure, drop_raw_*)
        - turbulence_window over [0, 2, 6]
        - lags_choice and precip_lags_choice over none/small/large
        - spatial_agg_method including "None" (no aggregation, per-farm
          features kept), "max", "n_turbines", "distance_n_turbines".

        Note: EMA also has `compute_wind_power_density` in its search space,
        but the flag is dead code in EMA's `_preprocess_location` — wind
        power density is actually gated on `compute_air_density`. We replicate
        EMA's runtime behaviour and omit the dead search dim.
        """
        return {
            "compute_air_density": trial.suggest_categorical(
                f"{prefix}_air_density", [True, False]
            ),
            "compute_air_density_moist": trial.suggest_categorical(
                f"{prefix}_density_moist", [True, False]
            ),
            "encode_wind_direction": trial.suggest_categorical(
                f"{prefix}_wind_dir", [True, False]
            ),
            "compute_wind_shear": trial.suggest_categorical(f"{prefix}_wind_shear", [True, False]),
            "compute_wind_ramp": trial.suggest_categorical(f"{prefix}_wind_ramp", [True, False]),
            "gust_factor": trial.suggest_categorical(f"{prefix}_gust", [True, False]),
            "dew_point_temperature": trial.suggest_categorical(
                f"{prefix}_dew_point", [True, False]
            ),
            "vapor_pressure": trial.suggest_categorical(f"{prefix}_vapor_pressure", [True, False]),
            "lags_choice": trial.suggest_categorical(f"{prefix}_lags", ["none", "small", "large"]),
            "precip_lags_choice": trial.suggest_categorical(
                f"{prefix}_precip_lags", ["none", "small", "large"]
            ),
            "turbulence_window": trial.suggest_categorical(f"{prefix}_turb_window", [0, 2, 6]),
            "drop_raw_main_features": trial.suggest_categorical(
                f"{prefix}_drop_main", [True, False]
            ),
            "drop_raw_wind_features": trial.suggest_categorical(
                f"{prefix}_drop_wind", [True, False]
            ),
            "spatial_agg_method": trial.suggest_categorical(
                f"{prefix}_spatial",
                [
                    "None",
                    "mean",
                    "max",
                    "idw",
                    "capacity",
                    "n_turbines",
                    "distance_capacity",
                    "distance_n_turbines",
                ],
            ),
        }
