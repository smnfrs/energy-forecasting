"""Spatial aggregation for multi-location weather features.

Collapses per-location columns into single aggregated features using
various weighting schemes (mean, IDW, capacity-weighted, etc.).

Ported from EMA's data_modules/feature_eng.py spatial aggregation methods.
"""

import numpy as np
import pandas as pd
from loguru import logger

from energy_forecasting.features.weather_physics import haversine_distance


def aggregate_locations(
    df: pd.DataFrame,
    suffixes: list[str],
    method: str,
    locations: list[dict],
) -> pd.DataFrame:
    """Aggregate per-location columns into single features.

    For each base feature that appears across multiple location suffixes,
    produces a single ``{base}_agg`` column.

    Parameters
    ----------
    df : DataFrame with per-location columns (e.g., ``wind_speed_100m_woff_enbw``)
    suffixes : Location suffixes to aggregate (e.g., ``["_woff_enbw", "_woff_bard"]``)
    method : Aggregation method — "mean", "max", "idw", "capacity", "population",
             "n_turbines", "n_panels", "energy", or compound like "distance_capacity"
    locations : Location metadata dicts (must have lat, lon, and weight field)

    Returns
    -------
    DataFrame with aggregated columns (``{base}_agg``) only.
    """
    if len(suffixes) <= 1:
        # Nothing to aggregate — rename by stripping suffix
        if suffixes:
            suffix = suffixes[0]
            rename = {c: c.replace(suffix, "_agg") for c in df.columns if c.endswith(suffix)}
            return df[list(rename.keys())].rename(columns=rename)
        return df

    # Group columns by base feature name
    feature_groups = _build_feature_map(df.columns, suffixes)

    if not feature_groups:
        logger.warning("No features found matching suffixes — skipping aggregation")
        return pd.DataFrame(index=df.index)

    # Compute weights
    weights = _compute_weights(method, locations, suffixes)

    result = pd.DataFrame(index=df.index)
    for base_name, col_map in feature_groups.items():
        sub_df = df[list(col_map.values())]
        if method in ("mean",):
            result[f"{base_name}_agg"] = sub_df.mean(axis=1)
        elif method in ("max",):
            result[f"{base_name}_agg"] = sub_df.max(axis=1)
        else:
            # Weighted average
            result[f"{base_name}_agg"] = _weighted_average(sub_df, col_map, weights)

    return result


def _build_feature_map(
    columns: pd.Index,
    suffixes: list[str],
) -> dict[str, dict[str, str]]:
    """Group columns by base feature name.

    Returns {base_name: {suffix: full_col_name}} for features that appear
    in at least 2 location suffixes.
    """
    groups: dict[str, dict[str, str]] = {}
    for col in columns:
        for suffix in suffixes:
            if col.endswith(suffix):
                base = col[: -len(suffix)]
                groups.setdefault(base, {})[suffix] = col
                break

    # Keep only features present in multiple locations
    return {base: cols for base, cols in groups.items() if len(cols) >= 2}


def _compute_weights(
    method: str,
    locations: list[dict],
    suffixes: list[str],
) -> dict[str, float]:
    """Compute per-suffix weights based on aggregation method.

    Returns {suffix: weight}.
    """
    loc_by_suffix = {loc["suffix"]: loc for loc in locations if loc.get("suffix") in suffixes}

    # Centroid for distance-based methods
    lats = [loc["lat"] for loc in loc_by_suffix.values()]
    lons = [loc["lon"] for loc in loc_by_suffix.values()]
    centroid_lat = np.mean(lats) if lats else 0
    centroid_lon = np.mean(lons) if lons else 0

    weights: dict[str, float] = {}

    for suffix in suffixes:
        loc = loc_by_suffix.get(suffix)
        if loc is None:
            weights[suffix] = 1.0
            continue

        dist = haversine_distance(loc["lat"], loc["lon"], centroid_lat, centroid_lon)
        dist = max(dist, 1.0)  # avoid division by zero

        if method == "idw":
            weights[suffix] = 1.0 / dist**2
        elif method == "capacity":
            weights[suffix] = loc.get("capacity", 1.0)
        elif method == "n_turbines":
            weights[suffix] = loc.get("n_turbines", 1.0)
        elif method == "n_panels":
            weights[suffix] = loc.get("n_panels", 1.0)
        elif method == "population":
            weights[suffix] = loc.get("population", 1.0)
        elif method == "energy":
            weights[suffix] = loc.get("total_energy_consumption", 1.0)
        elif method == "distance_capacity":
            weights[suffix] = loc.get("capacity", 1.0) / dist**2
        elif method == "distance_n_turbines":
            weights[suffix] = loc.get("n_turbines", 1.0) / dist**2
        elif method == "distance_n_panels":
            weights[suffix] = loc.get("n_panels", 1.0) / dist**2
        elif method == "distance_population":
            weights[suffix] = loc.get("population", 1.0) / dist**2
        elif method == "distance_energy":
            weights[suffix] = loc.get("total_energy_consumption", 1.0) / dist**2
        else:
            weights[suffix] = 1.0

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def _weighted_average(
    sub_df: pd.DataFrame,
    col_map: dict[str, str],
    weights: dict[str, float],
) -> pd.Series:
    """Weighted average across location columns."""
    weighted_sum = pd.Series(0.0, index=sub_df.index)
    total_weight = 0.0

    for suffix, col_name in col_map.items():
        w = weights.get(suffix, 0.0)
        weighted_sum += sub_df[col_name].fillna(0) * w
        total_weight += w

    if total_weight == 0:
        return sub_df.mean(axis=1)

    return weighted_sum / total_weight
