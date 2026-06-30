"""Load-related weather feature engineering.

Transforms raw weather variables into features relevant for electricity load
forecasting (heating/cooling degree hours, wind chill, humidex, etc.).

Ported from EMA's WeatherLoadFE class.
"""

import numpy as np
import pandas as pd

from energy_forecasting.features.spatial import aggregate_locations
from energy_forecasting.features.weather_physics import (
    air_density_dry,
    cooling_degree_hours,
    heating_degree_hours,
    humidex,
    wind_chill,
    wind_power_density,
)

DEFAULT_CONFIG: dict = {
    "compute_heating_degree_hours": True,
    "compute_cooling_degree_hours": True,
    "hdh_threshold": 18.0,
    "cdh_threshold": 22.0,
    "compute_dew_point_spread": True,
    "compute_temp_gradient": True,
    "compute_wind_chill": True,
    "compute_humidex": True,
    "compute_wind_components": True,
    "compute_wind_power_density": False,
    "compute_pressure_trend": True,
    "compute_wind_speed_gradient": False,
    "compute_air_density": False,
    "compute_rain_indicator": True,
    "compute_cloud_cover_fraction": True,
    "compute_effective_solar": True,
    "temp_lags_option": "small",
    "precip_lags_option": "none",
    "cloud_lags_option": "none",
    "rolling_temp_option": "short",
    "drop_basic_meteo_features": False,
    "drop_wind_meteo_features": False,
    "drop_rad_meteo_features": False,
    "spatial_agg_method": "population",
}

_TEMP_LAGS = {"none": [], "small": [1, 3], "medium": [1, 3, 6]}
_PRECIP_LAGS = {"none": [], "small": [1, 3], "medium": [1, 3, 6]}
_CLOUD_LAGS = {"none": [], "small": [1, 3], "medium": [1, 3, 6]}
_ROLLING_TEMP = {"none": [], "short": [3, 6], "long": [6, 12, 24]}


class WeatherLoadFE:
    """Load-related weather feature engineering for city locations."""

    def __init__(self, config: dict, locations: list[dict]):
        self.config = {**DEFAULT_CONFIG, **config}
        self.locations = locations
        self.suffixes = [loc["suffix"] for loc in locations]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        frames = [self._process_location(df, loc["suffix"]) for loc in self.locations]
        combined = pd.concat(frames, axis=1)

        method = self.config["spatial_agg_method"]
        if method != "None" and len(self.locations) > 1:
            combined = aggregate_locations(combined, self.suffixes, method, self.locations)

        return combined

    def _process_location(self, df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        cfg = self.config

        def col(name: str) -> str:
            return f"{name}{suffix}"

        def get(name: str) -> pd.Series | None:
            c = col(name)
            return df[c] if c in df.columns else None

        temp = get("temperature_2m")
        pressure = get("surface_pressure")
        humidity = get("relative_humidity_2m")
        ws10 = get("wind_speed_10m")
        wd10 = get("wind_direction_10m")
        cloud = get("cloud_cover")
        sw = get("shortwave_radiation")
        precip = get("precipitation")

        # Heating/cooling degree hours
        if cfg["compute_heating_degree_hours"] and temp is not None:
            result[f"hdh{suffix}"] = heating_degree_hours(temp, cfg["hdh_threshold"])
        if cfg["compute_cooling_degree_hours"] and temp is not None:
            result[f"cdh{suffix}"] = cooling_degree_hours(temp, cfg["cdh_threshold"])

        # Dew point spread
        if cfg["compute_dew_point_spread"] and temp is not None and humidity is not None:
            from energy_forecasting.features.weather_physics import dew_point_temperature

            dew = dew_point_temperature(temp, humidity)
            result[f"dew_point_spread{suffix}"] = temp - dew

        # Temperature gradient
        if cfg["compute_temp_gradient"] and temp is not None:
            result[f"temp_gradient{suffix}"] = temp.diff()

        # Wind chill
        if cfg["compute_wind_chill"] and temp is not None and ws10 is not None:
            result[f"wind_chill{suffix}"] = wind_chill(temp, ws10)

        # Humidex
        if cfg["compute_humidex"] and temp is not None and humidity is not None:
            result[f"humidex{suffix}"] = humidex(temp, humidity)

        # Wind components (U, V)
        if cfg["compute_wind_components"] and ws10 is not None and wd10 is not None:
            wd_rad = np.deg2rad(wd10)
            result[f"wind_u{suffix}"] = -ws10 * np.sin(wd_rad)
            result[f"wind_v{suffix}"] = -ws10 * np.cos(wd_rad)

        # Wind power density
        if cfg["compute_wind_power_density"] and ws10 is not None:
            density = (
                air_density_dry(pressure, temp)
                if pressure is not None and temp is not None
                else 1.225
            )
            result[f"wind_power_density{suffix}"] = wind_power_density(ws10, density)

        # Pressure trend
        if cfg["compute_pressure_trend"] and pressure is not None:
            result[f"pressure_trend{suffix}"] = pressure.diff()

        # Wind speed gradient
        if cfg["compute_wind_speed_gradient"] and ws10 is not None:
            result[f"wind_speed_gradient{suffix}"] = ws10.diff()

        # Air density
        if cfg["compute_air_density"] and pressure is not None and temp is not None:
            result[f"air_density{suffix}"] = air_density_dry(pressure, temp)

        # Rain indicator
        if cfg["compute_rain_indicator"] and precip is not None:
            result[f"rain_indicator{suffix}"] = (precip > 0).astype(float)

        # Cloud cover fraction
        if cfg["compute_cloud_cover_fraction"] and cloud is not None:
            result[f"cloud_fraction{suffix}"] = cloud / 100.0

        # Effective solar
        if cfg["compute_effective_solar"] and cloud is not None and sw is not None:
            cloud_frac = cloud / 100.0
            result[f"effective_solar{suffix}"] = (1.0 - cloud_frac) * sw

        # Temperature lags
        for lag in _TEMP_LAGS.get(cfg["temp_lags_option"], []):
            if temp is not None:
                result[f"temp_lag{lag}{suffix}"] = temp.shift(lag)

        # Precipitation lags
        for lag in _PRECIP_LAGS.get(cfg["precip_lags_option"], []):
            if precip is not None:
                result[f"precip_lag{lag}{suffix}"] = precip.shift(lag)

        # Cloud lags
        for lag in _CLOUD_LAGS.get(cfg["cloud_lags_option"], []):
            if cloud is not None:
                result[f"cloud_lag{lag}{suffix}"] = cloud.shift(lag)

        # Rolling temperature
        for window in _ROLLING_TEMP.get(cfg["rolling_temp_option"], []):
            if temp is not None:
                result[f"temp_rolling_mean_{window}{suffix}"] = temp.rolling(window=window).mean()

        # Raw features
        if not cfg["drop_basic_meteo_features"]:
            for raw in [
                "temperature_2m",
                "surface_pressure",
                "relative_humidity_2m",
                "precipitation",
            ]:
                rc = col(raw)
                if rc in df.columns:
                    result[rc] = df[rc]

        if not cfg["drop_wind_meteo_features"]:
            for raw in ["wind_speed_10m", "wind_direction_10m"]:
                rc = col(raw)
                if rc in df.columns:
                    result[rc] = df[rc]

        if not cfg["drop_rad_meteo_features"]:
            for raw in ["shortwave_radiation", "cloud_cover"]:
                rc = col(raw)
                if rc in df.columns:
                    result[rc] = df[rc]

        return result

    @staticmethod
    def suggest_optuna(trial, prefix: str = "load") -> dict:
        """Suggest hyperparameters for Optuna tuning.

        Matches EMA's WeatherLoadFE.selector_for_optuna search space:
        - 14 boolean toggles for derived features
        - 4 lag/rolling options
        - 3 raw-feature drops
        - spatial_agg_method including "None", "max", "energy", "distance_energy"

        Deviation from EMA: keeps `hdh_threshold` (15-20 °C) and
        `cdh_threshold` (20-26 °C) as float searches. EMA hardcodes 18 / 22
        and never searches them. Retained because load model already
        outperforms EMA on this configuration and the thresholds are
        physically meaningful tuning knobs.
        """
        return {
            "compute_heating_degree_hours": trial.suggest_categorical(
                f"{prefix}_hdh", [True, False]
            ),
            "compute_cooling_degree_hours": trial.suggest_categorical(
                f"{prefix}_cdh", [True, False]
            ),
            "hdh_threshold": trial.suggest_float(f"{prefix}_hdh_thresh", 15.0, 20.0),
            "cdh_threshold": trial.suggest_float(f"{prefix}_cdh_thresh", 20.0, 26.0),
            "compute_dew_point_spread": trial.suggest_categorical(
                f"{prefix}_dew_spread", [True, False]
            ),
            "compute_temp_gradient": trial.suggest_categorical(
                f"{prefix}_temp_grad", [True, False]
            ),
            "compute_wind_chill": trial.suggest_categorical(f"{prefix}_wind_chill", [True, False]),
            "compute_humidex": trial.suggest_categorical(f"{prefix}_humidex", [True, False]),
            "compute_wind_components": trial.suggest_categorical(
                f"{prefix}_wind_comp", [True, False]
            ),
            "compute_wind_power_density": trial.suggest_categorical(
                f"{prefix}_wpd", [True, False]
            ),
            "compute_wind_speed_gradient": trial.suggest_categorical(
                f"{prefix}_ws_grad", [True, False]
            ),
            "compute_pressure_trend": trial.suggest_categorical(
                f"{prefix}_pressure_trend", [True, False]
            ),
            "compute_air_density": trial.suggest_categorical(
                f"{prefix}_air_density", [True, False]
            ),
            "compute_rain_indicator": trial.suggest_categorical(f"{prefix}_rain", [True, False]),
            "compute_cloud_cover_fraction": trial.suggest_categorical(
                f"{prefix}_cloud_frac", [True, False]
            ),
            "compute_effective_solar": trial.suggest_categorical(
                f"{prefix}_eff_solar", [True, False]
            ),
            "temp_lags_option": trial.suggest_categorical(
                f"{prefix}_temp_lags", ["none", "small", "medium"]
            ),
            "precip_lags_option": trial.suggest_categorical(
                f"{prefix}_precip_lags", ["none", "small", "medium"]
            ),
            "cloud_lags_option": trial.suggest_categorical(
                f"{prefix}_cloud_lags", ["none", "small", "medium"]
            ),
            "rolling_temp_option": trial.suggest_categorical(
                f"{prefix}_rolling_temp", ["none", "short", "long"]
            ),
            "drop_basic_meteo_features": trial.suggest_categorical(
                f"{prefix}_drop_basic", [True, False]
            ),
            "drop_wind_meteo_features": trial.suggest_categorical(
                f"{prefix}_drop_wind", [True, False]
            ),
            "drop_rad_meteo_features": trial.suggest_categorical(
                f"{prefix}_drop_rad", [True, False]
            ),
            "spatial_agg_method": trial.suggest_categorical(
                f"{prefix}_spatial",
                [
                    "None",
                    "mean",
                    "max",
                    "idw",
                    "population",
                    "energy",
                    "distance_population",
                    "distance_energy",
                ],
            ),
        }
