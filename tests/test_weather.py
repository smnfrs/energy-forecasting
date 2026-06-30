"""Tests for data/weather.py — OpenMeteoSource with mocked data."""

import numpy as np
import pandas as pd
from energy_forecasting.data.weather import (
    ASSET_VARIABLES,
    PHYSICAL_LIMITS,
    _add_solar_columns,
    _validate_physical,
    load_locations,
)

# ── Physical validation ─────────────────────────────────────────────


def test_validate_physical_clips_temperature():
    values = np.array([-50.0, 25.0, 55.0])
    result = _validate_physical("temperature_2m", values)
    assert result[0] == -45.0  # clipped to min
    assert result[1] == 25.0  # unchanged
    assert result[2] == 50.0  # clipped to max


def test_validate_physical_clips_wind():
    values = np.array([-5.0, 100.0, 250.0])
    result = _validate_physical("wind_speed_100m", values)
    assert result[0] == 0.0
    assert result[1] == 100.0
    assert result[2] == 200.0


def test_validate_physical_passthrough_unknown():
    values = np.array([1.0, 2.0, 3.0])
    result = _validate_physical("unknown_variable", values)
    np.testing.assert_array_equal(result, values)


def test_physical_limits_cover_all_basic_vars():
    """All basic/wind/radiation variables should have limits defined."""
    for var_list in ASSET_VARIABLES.values():
        for var in var_list:
            assert var in PHYSICAL_LIMITS, f"Missing limit for {var}"


# ── Location loading ────────────────────────────────────────────────


def test_load_locations_offshore_tennet():
    locs = load_locations("offshore", "TenneT")
    assert len(locs) >= 4  # TenneT has 5 offshore farms
    for loc in locs:
        assert "lat" in loc
        assert "lon" in loc
        assert "suffix" in loc


def test_load_locations_offshore_50hertz():
    locs = load_locations("offshore", "50Hertz")
    assert len(locs) >= 2  # 50Hertz has Wikinger + Arkona


def test_load_locations_offshore_amprion_empty():
    locs = load_locations("offshore", "Amprion")
    assert len(locs) == 0  # No offshore farms for Amprion


def test_load_locations_onshore_has_locations():
    for tso in ["50Hertz", "Amprion", "TenneT", "TransnetBW"]:
        locs = load_locations("onshore", tso)
        assert len(locs) > 0, f"No onshore locations for {tso}"


def test_load_locations_cities_have_suffix():
    locs = load_locations("cities", "50Hertz")
    for loc in locs:
        assert loc["suffix"].startswith("_"), f"Suffix should start with _: {loc['suffix']}"


# ── Solar columns ───────────────────────────────────────────────────


def test_add_solar_columns():
    idx = pd.date_range("2024-06-21 12:00", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"temp": [20, 21, 22]}, index=idx)
    locations = [{"lat": 52.52, "lon": 13.41, "suffix": "_berlin"}]
    result = _add_solar_columns(df, locations)
    assert "solar_elevation_deg_berlin" in result.columns
    assert "solar_azimuth_deg_berlin" in result.columns
    # Midday in June in Berlin — sun should be well above horizon
    assert result["solar_elevation_deg_berlin"].iloc[0] > 30


# ── Asset variable sets ─────────────────────────────────────────────


def test_offshore_has_wind_vars():
    assert "wind_speed_100m" in ASSET_VARIABLES["offshore"]
    assert "shortwave_radiation" not in ASSET_VARIABLES["offshore"]


def test_solar_has_radiation_vars():
    assert "shortwave_radiation" in ASSET_VARIABLES["solar"]
    assert "wind_speed_100m" not in ASSET_VARIABLES["solar"]


def test_cities_has_all_vars():
    """Cities should have basic + wind + radiation."""
    cities_vars = ASSET_VARIABLES["cities"]
    assert "temperature_2m" in cities_vars
    assert "wind_speed_100m" in cities_vars
    assert "shortwave_radiation" in cities_vars
