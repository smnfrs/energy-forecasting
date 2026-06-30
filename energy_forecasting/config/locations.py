"""Location metadata for weather feature engineering.

Wraps eu_locations.json (ported from EMA's eu_locations.py) with
typed access. The JSON was converted from EMA's Python dict during
stage 2 data collection setup.
"""

import json
from functools import lru_cache
from typing import TypedDict

from energy_forecasting.config import LOCATIONS_DIR

# Type mapping from asset_type argument to JSON location type strings
_TYPE_MAP: dict[str, str] = {
    "offshore": "offshore wind farm",
    "onshore": "onshore wind farm",
    "solar": "solar farm",
    "cities": "city",
}

# Reverse: JSON key in de_locations -> location type string
_SECTION_TYPE: dict[str, str] = {
    "offshore_windfarms": "offshore wind farm",
    "onshore_windfarms": "onshore wind farm",
    "solarfarms": "solar farm",
    "cities": "city",
}


class LocationMeta(TypedDict, total=False):
    name: str
    label: str
    type: str  # "city", "onshore wind farm", "offshore wind farm", "solar farm"
    suffix: str  # e.g. "_woff_enbw", "_city_berlin"
    TSO: str
    lat: float
    lon: float
    capacity: float  # MW (wind/solar farms)
    n_turbines: int  # wind farms
    n_panels: int  # solar farms
    population: int  # cities
    total_energy_consumption: float  # GWh/year (cities)


@lru_cache
def load_locations() -> list[LocationMeta]:
    """Load all DE locations from eu_locations.json."""
    path = LOCATIONS_DIR / "eu_locations.json"
    with open(path) as f:
        data = json.load(f)

    de = data.get("de_locations", {})
    locations: list[LocationMeta] = []
    for section_key, type_str in _SECTION_TYPE.items():
        for loc in de.get(section_key, []):
            loc.setdefault("type", type_str)
            locations.append(loc)
    return locations


def locations_for_tso(tso: str, asset_type: str) -> list[LocationMeta]:
    """Filter locations by TSO and asset type.

    asset_type: "offshore", "onshore", "solar", "cities"
    Maps to location type strings in the JSON.
    """
    type_str = _TYPE_MAP.get(asset_type)
    if type_str is None:
        raise ValueError(f"Unknown asset_type {asset_type!r}, expected: {list(_TYPE_MAP)}")
    return [
        loc for loc in load_locations() if loc.get("TSO") == tso and loc.get("type") == type_str
    ]
