"""Solar power weather feature engineering.

Transforms raw weather variables (radiation, cloud cover, temperature, etc.)
into features relevant for solar power generation forecasting.

Ported from EMA's WeatherSolarPowerFE class.
"""

import numpy as np
import pandas as pd

from energy_forecasting.features.spatial import aggregate_locations
from energy_forecasting.features.weather_physics import (
    air_density_dry,
    air_density_moist,
    dew_point_temperature,
    vapor_pressure,
)

DEFAULT_CONFIG: dict = {
    "compute_cloud_cover_fraction": True,
    "compute_clear_sky_fraction": True,
    "compute_air_density": False,
    "compute_air_density_moist": False,
    "compute_direct_ratio": True,
    "compute_diffuse_ratio": True,
    "compute_dni_ratio": False,
    "compute_global_tilted_ratio": False,
    "use_solar_geometry": True,
    "dew_point_temperature": False,
    "vapor_pressure": False,
    "precip_lags_option": "none",
    "cloud_lags_option": "small",
    "shortwave_lags_option": "none",
    "drop_raw_solar_features": False,
    "drop_raw_features": False,
    "spatial_agg_method": "capacity",
}

_PRECIP_LAGS = {"none": [], "small": [1, 6], "large": [1, 6, 12, 24]}
_CLOUD_LAGS = {"none": [], "small": [1, 3], "medium": [1, 3, 6], "large": [1, 3, 6, 12]}
_SW_LAGS = {"none": [], "small": [1, 3], "medium": [1, 3, 6], "large": [1, 3, 6, 12]}


class WeatherSolarPowerFE:
    """Solar power weather feature engineering for a single TSO."""

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
        cloud = get("cloud_cover")
        sw = get("shortwave_radiation")
        direct = get("direct_radiation")
        diffuse = get("diffuse_radiation")
        dni = get("direct_normal_irradiance")
        gti = get("global_tilted_irradiance")

        # Cloud cover features
        if cfg["compute_cloud_cover_fraction"] and cloud is not None:
            result[f"cloud_fraction{suffix}"] = cloud / 100.0
        if cfg["compute_clear_sky_fraction"] and cloud is not None:
            result[f"clear_sky_fraction{suffix}"] = 1.0 - cloud / 100.0

        # Air density
        if cfg["compute_air_density"] and pressure is not None and temp is not None:
            result[f"air_density{suffix}"] = air_density_dry(pressure, temp)
        if (
            cfg["compute_air_density_moist"]
            and humidity is not None
            and temp is not None
            and pressure is not None
        ):
            result[f"air_density_moist{suffix}"] = air_density_moist(temp, pressure, humidity)

        # Radiation ratios (safe division — 0 when shortwave is 0)
        if sw is not None:
            sw_safe = sw.replace({0: np.nan})
            if cfg["compute_direct_ratio"] and direct is not None:
                result[f"direct_ratio{suffix}"] = (direct / sw_safe).fillna(0)
            if cfg["compute_diffuse_ratio"] and diffuse is not None:
                result[f"diffuse_ratio{suffix}"] = (diffuse / sw_safe).fillna(0)
            if cfg["compute_dni_ratio"] and dni is not None:
                result[f"dni_ratio{suffix}"] = (dni / sw_safe).fillna(0)
            if cfg["compute_global_tilted_ratio"] and gti is not None:
                result[f"gti_ratio{suffix}"] = (gti / sw_safe).fillna(0)

        # Solar geometry
        if cfg["use_solar_geometry"]:
            for geo in ["solar_elevation_deg", "solar_azimuth_deg"]:
                gc = col(geo)
                if gc in df.columns:
                    result[gc] = df[gc]

        # Dew point / vapor pressure
        if cfg["dew_point_temperature"] and temp is not None and humidity is not None:
            result[f"dew_point{suffix}"] = dew_point_temperature(temp, humidity)
        if cfg["vapor_pressure"] and temp is not None and humidity is not None:
            result[f"vapor_pressure{suffix}"] = vapor_pressure(temp, humidity)

        # Lags
        precip = get("precipitation")
        for lag in _PRECIP_LAGS.get(cfg["precip_lags_option"], []):
            if precip is not None:
                result[f"precipitation_lag{lag}{suffix}"] = precip.shift(lag)

        for lag in _CLOUD_LAGS.get(cfg["cloud_lags_option"], []):
            if cloud is not None:
                result[f"cloud_cover_lag{lag}{suffix}"] = cloud.shift(lag)

        for lag in _SW_LAGS.get(cfg["shortwave_lags_option"], []):
            if sw is not None:
                result[f"shortwave_lag{lag}{suffix}"] = sw.shift(lag)

        # Raw features
        if not cfg["drop_raw_solar_features"]:
            for raw in [
                "shortwave_radiation",
                "direct_radiation",
                "diffuse_radiation",
                "direct_normal_irradiance",
                "global_tilted_irradiance",
            ]:
                rc = col(raw)
                if rc in df.columns:
                    result[rc] = df[rc]

        if not cfg["drop_raw_features"]:
            for raw in [
                "temperature_2m",
                "surface_pressure",
                "relative_humidity_2m",
                "cloud_cover",
                "precipitation",
            ]:
                rc = col(raw)
                if rc in df.columns:
                    result[rc] = df[rc]

        return result

    @staticmethod
    def suggest_optuna(trial, prefix: str = "solar") -> dict:
        """Suggest hyperparameters for Optuna tuning.

        Matches EMA's WeatherSolarPowerFE.selector_for_optuna:
        - 11 boolean toggles (cloud_cover_fraction, clear_sky_fraction,
          air_density, air_density_moist, direct/diffuse/dni/global_tilted ratio,
          use_solar_geometry, dew_point, vapor_pressure)
        - 3 lag options (precip, cloud, shortwave) each over none/small/medium/large
          (precip is none/small/large in EMA)
        - 2 raw-feature drops
        - spatial_agg_method including "None" and panel-count weights.
        """
        return {
            "compute_cloud_cover_fraction": trial.suggest_categorical(
                f"{prefix}_cloud_frac", [True, False]
            ),
            "compute_clear_sky_fraction": trial.suggest_categorical(
                f"{prefix}_clear_sky", [True, False]
            ),
            "compute_air_density": trial.suggest_categorical(
                f"{prefix}_air_density", [True, False]
            ),
            "compute_air_density_moist": trial.suggest_categorical(
                f"{prefix}_density_moist", [True, False]
            ),
            "compute_direct_ratio": trial.suggest_categorical(
                f"{prefix}_direct_ratio", [True, False]
            ),
            "compute_diffuse_ratio": trial.suggest_categorical(
                f"{prefix}_diffuse_ratio", [True, False]
            ),
            "compute_dni_ratio": trial.suggest_categorical(f"{prefix}_dni_ratio", [True, False]),
            "compute_global_tilted_ratio": trial.suggest_categorical(
                f"{prefix}_gti_ratio", [True, False]
            ),
            "use_solar_geometry": trial.suggest_categorical(f"{prefix}_solar_geom", [True, False]),
            "dew_point_temperature": trial.suggest_categorical(
                f"{prefix}_dew_point", [True, False]
            ),
            "vapor_pressure": trial.suggest_categorical(f"{prefix}_vapor_pressure", [True, False]),
            "precip_lags_option": trial.suggest_categorical(
                f"{prefix}_precip_lags", ["none", "small", "large"]
            ),
            "cloud_lags_option": trial.suggest_categorical(
                f"{prefix}_cloud_lags", ["none", "small", "medium", "large"]
            ),
            "shortwave_lags_option": trial.suggest_categorical(
                f"{prefix}_sw_lags", ["none", "small", "medium", "large"]
            ),
            "drop_raw_solar_features": trial.suggest_categorical(
                f"{prefix}_drop_solar", [True, False]
            ),
            "drop_raw_features": trial.suggest_categorical(f"{prefix}_drop_raw", [True, False]),
            "spatial_agg_method": trial.suggest_categorical(
                f"{prefix}_spatial",
                [
                    "None",
                    "mean",
                    "max",
                    "idw",
                    "capacity",
                    "n_panels",
                    "distance_capacity",
                    "distance_n_panels",
                ],
            ),
        }
