"""Tests for weather feature engineering modules.

Covers:
- weather_physics: pure physics helper functions
- spatial: location aggregation (mean, capacity-weighted, single-location)
- weather_wind: WeatherWindPowerFE end-to-end
- weather_solar: WeatherSolarPowerFE end-to-end
- weather_load: WeatherLoadFE end-to-end
"""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.features.spatial import aggregate_locations
from energy_forecasting.features.weather_load import WeatherLoadFE
from energy_forecasting.features.weather_physics import (
    air_density_dry,
    air_density_moist,
    cooling_degree_hours,
    dew_point_temperature,
    haversine_distance,
    heating_degree_hours,
    wind_power_density,
)
from energy_forecasting.features.weather_solar import WeatherSolarPowerFE
from energy_forecasting.features.weather_wind import WeatherWindPowerFE

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def hourly_idx():
    """Short hourly DatetimeIndex for feature engineering tests."""
    return pd.date_range("2024-01-15", periods=48, freq="h", tz="UTC")


@pytest.fixture
def two_loc_wind_df(hourly_idx):
    """DataFrame with wind-relevant columns for two offshore locations."""
    rng = np.random.default_rng(0)
    n = len(hourly_idx)
    data = {}
    for suffix in ["_woff_a", "_woff_b"]:
        data[f"wind_speed_100m{suffix}"] = rng.uniform(5, 30, n)
        data[f"wind_speed_10m{suffix}"] = rng.uniform(3, 15, n)
        data[f"wind_direction_100m{suffix}"] = rng.uniform(0, 360, n)
        data[f"wind_gusts_10m{suffix}"] = rng.uniform(5, 40, n)
        data[f"temperature_2m{suffix}"] = rng.uniform(-5, 25, n)
        data[f"surface_pressure{suffix}"] = rng.uniform(990, 1030, n)
        data[f"relative_humidity_2m{suffix}"] = rng.uniform(40, 100, n)
        data[f"precipitation{suffix}"] = rng.exponential(0.5, n)
    return pd.DataFrame(data, index=hourly_idx)


@pytest.fixture
def wind_locations():
    """Two offshore location metadata dicts."""
    return [
        {"suffix": "_woff_a", "lat": 54.0, "lon": 6.5, "capacity": 400},
        {"suffix": "_woff_b", "lat": 54.5, "lon": 7.0, "capacity": 600},
    ]


@pytest.fixture
def single_loc_solar_df(hourly_idx):
    """DataFrame with solar-relevant columns for one location."""
    rng = np.random.default_rng(1)
    n = len(hourly_idx)
    suffix = "_sol_a"
    return pd.DataFrame(
        {
            f"temperature_2m{suffix}": rng.uniform(0, 30, n),
            f"surface_pressure{suffix}": rng.uniform(990, 1030, n),
            f"relative_humidity_2m{suffix}": rng.uniform(30, 95, n),
            f"cloud_cover{suffix}": rng.uniform(0, 100, n),
            f"shortwave_radiation{suffix}": rng.uniform(0, 800, n),
            f"direct_radiation{suffix}": rng.uniform(0, 600, n),
            f"diffuse_radiation{suffix}": rng.uniform(0, 300, n),
            f"precipitation{suffix}": rng.exponential(0.3, n),
            f"solar_elevation_deg{suffix}": rng.uniform(-10, 60, n),
            f"solar_azimuth_deg{suffix}": rng.uniform(0, 360, n),
        },
        index=hourly_idx,
    )


@pytest.fixture
def solar_locations_single():
    return [{"suffix": "_sol_a", "lat": 51.0, "lon": 10.0, "capacity": 500}]


@pytest.fixture
def single_loc_load_df(hourly_idx):
    """DataFrame with load-relevant columns for one city."""
    rng = np.random.default_rng(2)
    n = len(hourly_idx)
    suffix = "_berlin"
    return pd.DataFrame(
        {
            f"temperature_2m{suffix}": rng.uniform(-10, 35, n),
            f"surface_pressure{suffix}": rng.uniform(990, 1030, n),
            f"relative_humidity_2m{suffix}": rng.uniform(30, 95, n),
            f"wind_speed_10m{suffix}": rng.uniform(0, 60, n),
            f"wind_direction_10m{suffix}": rng.uniform(0, 360, n),
            f"cloud_cover{suffix}": rng.uniform(0, 100, n),
            f"shortwave_radiation{suffix}": rng.uniform(0, 600, n),
            f"precipitation{suffix}": rng.exponential(0.5, n),
        },
        index=hourly_idx,
    )


@pytest.fixture
def load_locations_single():
    return [{"suffix": "_berlin", "lat": 52.52, "lon": 13.41, "population": 3_600_000}]


# ═══════════════════════════════════════════════════════════════════
# 1. Physics helpers
# ═══════════════════════════════════════════════════════════════════


class TestAirDensityDry:
    def test_standard_conditions(self):
        """ISA standard atmosphere: 15 C, 1013.25 hPa -> ~1.225 kg/m3."""
        temp = pd.Series([15.0])
        pressure = pd.Series([1013.25])
        rho = air_density_dry(pressure, temp)
        assert rho.iloc[0] == pytest.approx(1.225, abs=0.005)

    @pytest.mark.parametrize(
        "temp_c, pressure_hpa",
        [(-20.0, 1013.25), (0.0, 1013.25), (40.0, 1013.25)],
        ids=["cold", "freezing", "hot"],
    )
    def test_density_decreases_with_temperature(self, temp_c, pressure_hpa):
        """At fixed pressure, warmer air is less dense."""
        temps = pd.Series([temp_c, temp_c + 10])
        pressures = pd.Series([pressure_hpa, pressure_hpa])
        rho = air_density_dry(pressures, temps)
        assert rho.iloc[0] > rho.iloc[1]

    def test_density_increases_with_pressure(self):
        """At fixed temperature, higher pressure means denser air."""
        temp = pd.Series([15.0, 15.0])
        pressure = pd.Series([1000.0, 1020.0])
        rho = air_density_dry(pressure, temp)
        assert rho.iloc[1] > rho.iloc[0]


class TestAirDensityMoist:
    def test_moist_less_than_dry(self):
        """Moist air is less dense than dry air at the same T and P."""
        temp = pd.Series([20.0])
        pressure = pd.Series([1013.25])
        humidity = pd.Series([80.0])  # 80 % RH
        rho_dry = air_density_dry(pressure, temp)
        rho_moist = air_density_moist(temp, pressure, humidity)
        assert rho_moist.iloc[0] < rho_dry.iloc[0]

    def test_zero_humidity_approaches_dry(self):
        """At 0 % RH, moist density should be very close to dry density."""
        temp = pd.Series([15.0])
        pressure = pd.Series([1013.25])
        humidity = pd.Series([0.1])  # clip floor in dew point, use near-zero
        rho_dry = air_density_dry(pressure, temp)
        rho_moist = air_density_moist(temp, pressure, humidity)
        # Should be very close (within 0.1 %)
        assert rho_moist.iloc[0] == pytest.approx(rho_dry.iloc[0], rel=0.001)


class TestWindPowerDensity:
    def test_zero_wind(self):
        """Zero wind speed produces zero power density."""
        ws = pd.Series([0.0])
        rho = pd.Series([1.225])
        wpd = wind_power_density(ws, rho)
        assert wpd.iloc[0] == 0.0

    def test_proportional_to_v_cubed(self):
        """Doubling wind speed should give 8x the power density."""
        rho = pd.Series([1.225, 1.225])
        ws = pd.Series([10.0, 20.0])  # km/h
        wpd = wind_power_density(ws, rho)
        ratio = wpd.iloc[1] / wpd.iloc[0]
        assert ratio == pytest.approx(8.0, rel=1e-6)

    def test_known_value(self):
        """10 m/s = 36 km/h at standard density: P/A = 0.5 * 1.225 * 10^3 = 612.5 W/m2."""
        ws_kmh = pd.Series([36.0])
        rho = pd.Series([1.225])
        wpd = wind_power_density(ws_kmh, rho)
        assert wpd.iloc[0] == pytest.approx(612.5, abs=1.0)


class TestDewPointTemperature:
    def test_dew_point_leq_temperature(self):
        """Dew point can never exceed the actual temperature."""
        rng = np.random.default_rng(99)
        temps = pd.Series(rng.uniform(-10, 40, 200))
        humidities = pd.Series(rng.uniform(5, 100, 200))
        dew = dew_point_temperature(temps, humidities)
        assert (dew <= temps + 0.01).all()  # small tolerance for numerics

    def test_100pct_humidity(self):
        """At 100 % RH, dew point equals the temperature (Magnus approx)."""
        temps = pd.Series([10.0, 20.0, 30.0])
        rh = pd.Series([100.0, 100.0, 100.0])
        dew = dew_point_temperature(temps, rh)
        np.testing.assert_allclose(dew.values, temps.values, atol=0.5)


class TestHeatingDegreeHours:
    @pytest.mark.parametrize(
        "temp, expected",
        [(10.0, 8.0), (18.0, 0.0), (25.0, 0.0), (0.0, 18.0)],
        ids=["below", "at_threshold", "above", "deep_cold"],
    )
    def test_hdh_values(self, temp, expected):
        result = heating_degree_hours(pd.Series([temp]))
        assert result.iloc[0] == pytest.approx(expected)

    def test_custom_threshold(self):
        result = heating_degree_hours(pd.Series([10.0]), threshold=15.0)
        assert result.iloc[0] == pytest.approx(5.0)


class TestCoolingDegreeHours:
    @pytest.mark.parametrize(
        "temp, expected",
        [(10.0, 0.0), (22.0, 0.0), (30.0, 8.0), (35.0, 13.0)],
        ids=["cold", "at_threshold", "warm", "hot"],
    )
    def test_cdh_values(self, temp, expected):
        result = cooling_degree_hours(pd.Series([temp]))
        assert result.iloc[0] == pytest.approx(expected)

    def test_custom_threshold(self):
        result = cooling_degree_hours(pd.Series([28.0]), threshold=25.0)
        assert result.iloc[0] == pytest.approx(3.0)


class TestHaversineDistance:
    def test_berlin_to_munich(self):
        """Berlin (52.52, 13.41) to Munich (48.14, 11.58) ~ 504 km."""
        dist = haversine_distance(52.52, 13.41, 48.14, 11.58)
        assert dist == pytest.approx(504.0, abs=5.0)

    def test_same_point(self):
        assert haversine_distance(50.0, 10.0, 50.0, 10.0) == pytest.approx(0.0)

    def test_symmetry(self):
        d1 = haversine_distance(52.0, 13.0, 48.0, 11.0)
        d2 = haversine_distance(48.0, 11.0, 52.0, 13.0)
        assert d1 == pytest.approx(d2)


# ═══════════════════════════════════════════════════════════════════
# 2. Spatial aggregation
# ═══════════════════════════════════════════════════════════════════


class TestAggregateLocationsMean:
    def test_mean_of_two_locations(self):
        """Mean aggregation should average columns with matching base names."""
        idx = pd.RangeIndex(5)
        df = pd.DataFrame(
            {
                "wind_speed_100m_loc_a": [10.0, 20.0, 30.0, 40.0, 50.0],
                "wind_speed_100m_loc_b": [20.0, 30.0, 40.0, 50.0, 60.0],
            },
            index=idx,
        )
        suffixes = ["_loc_a", "_loc_b"]
        locations = [
            {"suffix": "_loc_a", "lat": 50.0, "lon": 10.0},
            {"suffix": "_loc_b", "lat": 51.0, "lon": 11.0},
        ]
        result = aggregate_locations(df, suffixes, "mean", locations)
        assert "wind_speed_100m_agg" in result.columns
        np.testing.assert_allclose(
            result["wind_speed_100m_agg"].values,
            [15.0, 25.0, 35.0, 45.0, 55.0],
        )


class TestAggregateLocationsCapacity:
    def test_capacity_weighted(self):
        """Capacity-weighted aggregation: weight proportional to capacity."""
        idx = pd.RangeIndex(3)
        df = pd.DataFrame(
            {
                "temp_loc_a": [10.0, 20.0, 30.0],
                "temp_loc_b": [20.0, 30.0, 40.0],
            },
            index=idx,
        )
        suffixes = ["_loc_a", "_loc_b"]
        locations = [
            {"suffix": "_loc_a", "lat": 50.0, "lon": 10.0, "capacity": 100},
            {"suffix": "_loc_b", "lat": 51.0, "lon": 11.0, "capacity": 300},
        ]
        result = aggregate_locations(df, suffixes, "capacity", locations)
        # Weights: 100/(100+300) = 0.25, 300/(100+300) = 0.75
        expected = 0.25 * np.array([10.0, 20.0, 30.0]) + 0.75 * np.array([20.0, 30.0, 40.0])
        np.testing.assert_allclose(result["temp_agg"].values, expected, atol=1e-6)


class TestAggregateLocationsSingle:
    def test_single_location_renames(self):
        """With one location, columns are just renamed from suffix to _agg."""
        idx = pd.RangeIndex(3)
        df = pd.DataFrame(
            {"wind_speed_100m_loc_a": [10.0, 20.0, 30.0]},
            index=idx,
        )
        result = aggregate_locations(
            df,
            suffixes=["_loc_a"],
            method="mean",
            locations=[{"suffix": "_loc_a", "lat": 50.0, "lon": 10.0}],
        )
        assert "wind_speed_100m_agg" in result.columns
        assert "wind_speed_100m_loc_a" not in result.columns
        np.testing.assert_array_equal(result["wind_speed_100m_agg"].values, [10.0, 20.0, 30.0])


# ═══════════════════════════════════════════════════════════════════
# 3. WeatherWindPowerFE
# ═══════════════════════════════════════════════════════════════════


class TestWeatherWindPowerFE:
    def test_wind_power_density_column(self, two_loc_wind_df, wind_locations):
        """wind_power_density column should exist for each location before aggregation."""
        config = {
            "spatial_agg_method": "None",  # skip aggregation to inspect per-location
            "lags_choice": "none",
            "precip_lags_choice": "none",
        }
        fe = WeatherWindPowerFE(config, wind_locations)
        result = fe(two_loc_wind_df)
        assert "wind_power_density_woff_a" in result.columns
        assert "wind_power_density_woff_b" in result.columns
        # Values should be non-negative
        assert (result["wind_power_density_woff_a"] >= 0).all()

    def test_wind_direction_sin_cos(self, two_loc_wind_df, wind_locations):
        """Wind direction encoding should produce sin and cos columns in [-1, 1]."""
        config = {
            "encode_wind_direction": True,
            "spatial_agg_method": "None",
            "lags_choice": "none",
            "precip_lags_choice": "none",
        }
        fe = WeatherWindPowerFE(config, wind_locations)
        result = fe(two_loc_wind_df)
        for suffix in ["_woff_a", "_woff_b"]:
            sin_col = f"wind_dir_sin{suffix}"
            cos_col = f"wind_dir_cos{suffix}"
            assert sin_col in result.columns
            assert cos_col in result.columns
            assert result[sin_col].between(-1, 1).all()
            assert result[cos_col].between(-1, 1).all()

    def test_spatial_aggregation_reduces_columns(self, two_loc_wind_df, wind_locations):
        """After capacity aggregation, per-location columns become _agg columns."""
        config = {
            "spatial_agg_method": "capacity",
            "lags_choice": "none",
            "precip_lags_choice": "none",
            "drop_raw_main_features": True,
            "drop_raw_wind_features": True,
        }
        fe = WeatherWindPowerFE(config, wind_locations)
        result = fe(two_loc_wind_df)
        # Should have _agg columns, not per-location ones
        agg_cols = [c for c in result.columns if c.endswith("_agg")]
        loc_cols = [c for c in result.columns if c.endswith("_woff_a") or c.endswith("_woff_b")]
        assert len(agg_cols) > 0
        assert len(loc_cols) == 0

    def test_no_direction_encoding_when_disabled(self, two_loc_wind_df, wind_locations):
        """encode_wind_direction=False should omit sin/cos columns."""
        config = {
            "encode_wind_direction": False,
            "spatial_agg_method": "None",
            "lags_choice": "none",
            "precip_lags_choice": "none",
        }
        fe = WeatherWindPowerFE(config, wind_locations)
        result = fe(two_loc_wind_df)
        sin_cos_cols = [c for c in result.columns if "wind_dir_sin" in c or "wind_dir_cos" in c]
        assert len(sin_cos_cols) == 0


# ═══════════════════════════════════════════════════════════════════
# 4. WeatherSolarPowerFE
# ═══════════════════════════════════════════════════════════════════


class TestWeatherSolarPowerFE:
    def test_cloud_fraction(self, single_loc_solar_df, solar_locations_single):
        """cloud_fraction should be cloud_cover / 100."""
        config = {
            "compute_cloud_cover_fraction": True,
            "spatial_agg_method": "None",
            "cloud_lags_option": "none",
            "shortwave_lags_option": "none",
            "precip_lags_option": "none",
        }
        fe = WeatherSolarPowerFE(config, solar_locations_single)
        result = fe(single_loc_solar_df)
        expected = single_loc_solar_df["cloud_cover_sol_a"] / 100.0
        np.testing.assert_allclose(
            result["cloud_fraction_sol_a"].values,
            expected.values,
            atol=1e-10,
        )

    def test_radiation_ratios(self, single_loc_solar_df, solar_locations_single):
        """direct_ratio = direct / shortwave (0 where shortwave is 0)."""
        config = {
            "compute_direct_ratio": True,
            "compute_diffuse_ratio": True,
            "spatial_agg_method": "None",
            "cloud_lags_option": "none",
            "shortwave_lags_option": "none",
            "precip_lags_option": "none",
        }
        fe = WeatherSolarPowerFE(config, solar_locations_single)
        result = fe(single_loc_solar_df)
        assert "direct_ratio_sol_a" in result.columns
        assert "diffuse_ratio_sol_a" in result.columns
        # Ratios should be non-negative
        assert (result["direct_ratio_sol_a"] >= 0).all()
        assert (result["diffuse_ratio_sol_a"] >= 0).all()

    def test_solar_geometry_preserved(self, single_loc_solar_df, solar_locations_single):
        """When use_solar_geometry=True, elevation and azimuth columns pass through."""
        config = {
            "use_solar_geometry": True,
            "spatial_agg_method": "None",
            "cloud_lags_option": "none",
            "shortwave_lags_option": "none",
            "precip_lags_option": "none",
        }
        fe = WeatherSolarPowerFE(config, solar_locations_single)
        result = fe(single_loc_solar_df)
        assert "solar_elevation_deg_sol_a" in result.columns
        assert "solar_azimuth_deg_sol_a" in result.columns
        np.testing.assert_array_equal(
            result["solar_elevation_deg_sol_a"].values,
            single_loc_solar_df["solar_elevation_deg_sol_a"].values,
        )

    def test_solar_geometry_omitted_when_disabled(
        self, single_loc_solar_df, solar_locations_single
    ):
        config = {
            "use_solar_geometry": False,
            "spatial_agg_method": "None",
            "cloud_lags_option": "none",
            "shortwave_lags_option": "none",
            "precip_lags_option": "none",
        }
        fe = WeatherSolarPowerFE(config, solar_locations_single)
        result = fe(single_loc_solar_df)
        geo_cols = [c for c in result.columns if "solar_elevation" in c or "solar_azimuth" in c]
        assert len(geo_cols) == 0


# ═══════════════════════════════════════════════════════════════════
# 5. WeatherLoadFE
# ═══════════════════════════════════════════════════════════════════


class TestWeatherLoadFE:
    def test_hdh_cdh_computed(self, single_loc_load_df, load_locations_single):
        """HDH and CDH columns should exist and be non-negative."""
        config = {
            "compute_heating_degree_hours": True,
            "compute_cooling_degree_hours": True,
            "spatial_agg_method": "None",
            "temp_lags_option": "none",
            "precip_lags_option": "none",
            "cloud_lags_option": "none",
            "rolling_temp_option": "none",
        }
        fe = WeatherLoadFE(config, load_locations_single)
        result = fe(single_loc_load_df)
        assert "hdh_berlin" in result.columns
        assert "cdh_berlin" in result.columns
        assert (result["hdh_berlin"] >= 0).all()
        assert (result["cdh_berlin"] >= 0).all()

    def test_hdh_cdh_correctness(self, hourly_idx):
        """Verify HDH/CDH against known temperatures."""
        suffix = "_test"
        temps = [10.0, 18.0, 22.0, 30.0]
        n = len(temps)
        idx = hourly_idx[:n]
        df = pd.DataFrame(
            {
                f"temperature_2m{suffix}": temps,
                f"surface_pressure{suffix}": [1013.25] * n,
                f"relative_humidity_2m{suffix}": [60.0] * n,
                f"wind_speed_10m{suffix}": [10.0] * n,
                f"wind_direction_10m{suffix}": [180.0] * n,
                f"cloud_cover{suffix}": [50.0] * n,
                f"shortwave_radiation{suffix}": [200.0] * n,
                f"precipitation{suffix}": [0.0] * n,
            },
            index=idx,
        )
        locs = [{"suffix": suffix, "lat": 50.0, "lon": 10.0, "population": 1_000_000}]
        config = {
            "hdh_threshold": 18.0,
            "cdh_threshold": 22.0,
            "spatial_agg_method": "None",
            "temp_lags_option": "none",
            "precip_lags_option": "none",
            "cloud_lags_option": "none",
            "rolling_temp_option": "none",
        }
        fe = WeatherLoadFE(config, locs)
        result = fe(df)
        # 10 C: HDH=8, CDH=0; 18 C: HDH=0, CDH=0; 22 C: HDH=0, CDH=0; 30 C: HDH=0, CDH=8
        np.testing.assert_allclose(result[f"hdh{suffix}"].values, [8.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result[f"cdh{suffix}"].values, [0.0, 0.0, 0.0, 8.0])

    def test_wind_chill_computed(self, single_loc_load_df, load_locations_single):
        """wind_chill column should exist when enabled."""
        config = {
            "compute_wind_chill": True,
            "spatial_agg_method": "None",
            "temp_lags_option": "none",
            "precip_lags_option": "none",
            "cloud_lags_option": "none",
            "rolling_temp_option": "none",
        }
        fe = WeatherLoadFE(config, load_locations_single)
        result = fe(single_loc_load_df)
        assert "wind_chill_berlin" in result.columns
        # Wind chill should be lower than or comparable to actual temperature
        # (formula can exceed temp in some edge cases with low wind, so just check it exists)
        assert not result["wind_chill_berlin"].isna().all()

    def test_rain_indicator_is_binary(self, single_loc_load_df, load_locations_single):
        """rain_indicator should be 0.0 or 1.0 only."""
        config = {
            "compute_rain_indicator": True,
            "spatial_agg_method": "None",
            "temp_lags_option": "none",
            "precip_lags_option": "none",
            "cloud_lags_option": "none",
            "rolling_temp_option": "none",
        }
        fe = WeatherLoadFE(config, load_locations_single)
        result = fe(single_loc_load_df)
        assert "rain_indicator_berlin" in result.columns
        unique_vals = set(result["rain_indicator_berlin"].unique())
        assert unique_vals.issubset({0.0, 1.0})

    def test_rain_indicator_matches_precipitation(self, hourly_idx):
        """rain_indicator should be 1 when precipitation > 0, else 0."""
        suffix = "_test"
        precip_vals = [0.0, 0.5, 0.0, 3.2]
        n = len(precip_vals)
        idx = hourly_idx[:n]
        df = pd.DataFrame(
            {
                f"temperature_2m{suffix}": [15.0] * n,
                f"surface_pressure{suffix}": [1013.25] * n,
                f"relative_humidity_2m{suffix}": [60.0] * n,
                f"wind_speed_10m{suffix}": [10.0] * n,
                f"wind_direction_10m{suffix}": [180.0] * n,
                f"cloud_cover{suffix}": [50.0] * n,
                f"shortwave_radiation{suffix}": [200.0] * n,
                f"precipitation{suffix}": precip_vals,
            },
            index=idx,
        )
        locs = [{"suffix": suffix, "lat": 50.0, "lon": 10.0, "population": 500_000}]
        config = {
            "compute_rain_indicator": True,
            "spatial_agg_method": "None",
            "temp_lags_option": "none",
            "precip_lags_option": "none",
            "cloud_lags_option": "none",
            "rolling_temp_option": "none",
        }
        fe = WeatherLoadFE(config, locs)
        result = fe(df)
        np.testing.assert_array_equal(
            result[f"rain_indicator{suffix}"].values,
            [0.0, 1.0, 0.0, 1.0],
        )
