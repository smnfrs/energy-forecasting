"""Audit which Optuna search-space options were actually selected as winners
across the v4 (2026-05-06 18:18 CEST onward) base-model runs.

For each (target, region, model_type), pulls the latest finished base run's
``optuna/best_config.json`` artifact, extracts the winning FE config + dataset
params + model params, and tallies how often each value of each search dim
appeared in the winners.

Outputs (written to notebooks/phase_a_review/06_search_space_audit.txt):
- Per-FE-class selection histograms for all categorical/boolean dims
- Continuous-dim summaries (min / median / max chosen)
- Flagged "removal candidates" — options whose selection rate is 0 / 47

Usage:
    python scripts/audit_search_space.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient

from energy_forecasting.config.modeling import EXPERIMENTS, GEN_LOAD_TARGETS

V4_START_UTC = pd.Timestamp("2026-05-06 16:00:00+0000")  # 18:18 CEST
V4_END_UTC = pd.Timestamp("2026-05-07 03:00:00+0000")

GEN_TARGETS = ["wind_onshore", "wind_offshore", "solar", "load", "gen_load_diff"]
TARGET_TO_EXP = {
    "wind_onshore": "gen_wind_onshore",
    "wind_offshore": "gen_wind_offshore",
    "solar": "gen_solar",
    "load": "gen_load",
    "gen_load_diff": "gen_gen_load_diff",
}
MODEL_TYPES = ["LGBMRegressor", "XGBRegressor", "ElasticNet"]

# Group targets by FE class so we tally per-class, not blended together.
# (Searching a categorical that doesn't exist in some target's space is fine —
# only target/regions whose runs actually have the key contribute.)
FE_CLASSES = {
    "wind": ["wind_onshore", "wind_offshore"],
    "solar": ["solar"],
    "load": ["load", "gen_load_diff"],  # gen_load_diff also uses load FE for weather
}


def _v4_runs(client: MlflowClient, target: str, region: str, model_type: str):
    exp_path = EXPERIMENTS[TARGET_TO_EXP[target]]
    exp = client.get_experiment_by_name(exp_path)
    if exp is None:
        return []
    fv = f"optuna_{model_type.lower()}"
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=(
            f"tags.target='{target}' AND tags.region='{region}' "
            f"AND tags.feature_version='{fv}' AND attributes.status='FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    out = []
    for r in runs:
        st = pd.Timestamp(r.info.start_time, unit="ms", tz="UTC")
        if V4_START_UTC <= st <= V4_END_UTC:
            out.append(r)
    return out


def main():
    client = MlflowClient()
    # winners[fe_class][param_name] -> Counter / list
    winners_cat: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    winners_cont: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    # Also tally model + dataset params globally
    global_model_cat: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    global_model_cont: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    global_ds_cat: dict[str, Counter] = defaultdict(Counter)
    global_ds_cont: dict[str, list] = defaultdict(list)

    n_collected = 0
    misses: list[str] = []

    for target in GEN_TARGETS:
        regions = GEN_LOAD_TARGETS[target]["regions"]
        fe_class = next(c for c, ts in FE_CLASSES.items() if target in ts)
        for region in regions:
            for model_type in MODEL_TYPES:
                runs = _v4_runs(client, target, region, model_type)
                if not runs:
                    misses.append(f"{target}/{region}/{model_type}: no v4 run found")
                    continue
                run = runs[0]
                try:
                    art_path = client.download_artifacts(
                        run.info.run_id, "optuna/best_config.json"
                    )
                except Exception as e:
                    misses.append(
                        f"{target}/{region}/{model_type}: artifact download failed ({e})"
                    )
                    continue
                with open(art_path) as f:
                    best = json.load(f)

                wcfg = best.get("weather_config", {})
                for k, v in wcfg.items():
                    if isinstance(v, bool):
                        winners_cat[fe_class][k][str(v)] += 1
                    elif isinstance(v, (int, float)):
                        winners_cont[fe_class][k].append(float(v))
                    else:
                        winners_cat[fe_class][k][str(v)] += 1

                ds = best.get("dataset_params", {})
                for k, v in ds.items():
                    if isinstance(v, bool):
                        global_ds_cat[k][str(v)] += 1
                    elif isinstance(v, (int, float)):
                        global_ds_cont[k].append(float(v))
                    else:
                        global_ds_cat[k][str(v)] += 1

                mp = best.get("model_params", {})
                for k, v in mp.items():
                    if isinstance(v, bool):
                        global_model_cat[model_type][k][str(v)] += 1
                    elif isinstance(v, (int, float)):
                        global_model_cont[model_type][k].append(float(v))
                    else:
                        global_model_cat[model_type][k][str(v)] += 1

                n_collected += 1

    # Per-FE-class expected option set (declared in suggest_optuna). We hardcode
    # to detect "never-selected" options: any allowed value with 0 selections.
    EXPECTED = {
        "wind": {
            "compute_air_density": ["True", "False"],
            "compute_air_density_moist": ["True", "False"],
            "encode_wind_direction": ["True", "False"],
            "compute_wind_shear": ["True", "False"],
            "compute_wind_ramp": ["True", "False"],
            "gust_factor": ["True", "False"],
            "dew_point_temperature": ["True", "False"],
            "vapor_pressure": ["True", "False"],
            "lags_choice": ["none", "small", "large"],
            "precip_lags_choice": ["none", "small", "large"],
            "turbulence_window": [],  # continuous-ish (0/2/6)
            "drop_raw_main_features": ["True", "False"],
            "drop_raw_wind_features": ["True", "False"],
            "spatial_agg_method": [
                "None", "mean", "max", "idw", "capacity",
                "n_turbines", "distance_capacity", "distance_n_turbines",
            ],
        },
        "solar": {
            "compute_cloud_cover_fraction": ["True", "False"],
            "compute_clear_sky_fraction": ["True", "False"],
            "compute_air_density": ["True", "False"],
            "compute_air_density_moist": ["True", "False"],
            "compute_direct_ratio": ["True", "False"],
            "compute_diffuse_ratio": ["True", "False"],
            "compute_dni_ratio": ["True", "False"],
            "compute_global_tilted_ratio": ["True", "False"],
            "use_solar_geometry": ["True", "False"],
            "dew_point_temperature": ["True", "False"],
            "vapor_pressure": ["True", "False"],
            "precip_lags_option": ["none", "small", "large"],
            "cloud_lags_option": ["none", "small", "medium", "large"],
            "shortwave_lags_option": ["none", "small", "medium", "large"],
            "drop_raw_solar_features": ["True", "False"],
            "drop_raw_features": ["True", "False"],
            "spatial_agg_method": [
                "None", "mean", "max", "idw", "capacity",
                "n_panels", "distance_capacity", "distance_n_panels",
            ],
        },
        "load": {
            "compute_heating_degree_hours": ["True", "False"],
            "compute_cooling_degree_hours": ["True", "False"],
            "compute_dew_point_spread": ["True", "False"],
            "compute_temp_gradient": ["True", "False"],
            "compute_wind_chill": ["True", "False"],
            "compute_humidex": ["True", "False"],
            "compute_wind_components": ["True", "False"],
            "compute_wind_power_density": ["True", "False"],
            "compute_wind_speed_gradient": ["True", "False"],
            "compute_pressure_trend": ["True", "False"],
            "compute_air_density": ["True", "False"],
            "compute_rain_indicator": ["True", "False"],
            "compute_cloud_cover_fraction": ["True", "False"],
            "compute_effective_solar": ["True", "False"],
            "temp_lags_option": ["none", "small", "medium"],
            "precip_lags_option": ["none", "small", "medium"],
            "cloud_lags_option": ["none", "small", "medium"],
            "rolling_temp_option": ["none", "short", "long"],
            "drop_basic_meteo_features": ["True", "False"],
            "drop_wind_meteo_features": ["True", "False"],
            "drop_rad_meteo_features": ["True", "False"],
            "spatial_agg_method": [
                "None", "mean", "max", "idw", "population",
                "energy", "distance_population", "distance_energy",
            ],
        },
    }

    out_lines = []
    out_lines.append("=" * 100)
    out_lines.append(
        f"v4 search-space audit — {n_collected} winning base configs across "
        f"{len(GEN_TARGETS)} targets × {len(MODEL_TYPES)} model types"
    )
    out_lines.append("=" * 100)
    if misses:
        out_lines.append("\nMisses:")
        for m in misses:
            out_lines.append(f"  {m}")

    for fe_class in ["wind", "solar", "load"]:
        n_runs_class = sum(
            sum(c.values()) for c in winners_cat[fe_class].values()
        ) // max(1, len(winners_cat[fe_class]))
        out_lines.append("\n" + "─" * 100)
        out_lines.append(f"FE class: {fe_class}  ({n_runs_class} contributing runs)")
        out_lines.append("─" * 100)

        # Categorical/boolean dims
        for param, expected_vals in EXPECTED[fe_class].items():
            counter = winners_cat[fe_class].get(param, Counter())
            cont_vals = winners_cont[fe_class].get(param, [])
            total = sum(counter.values())
            if expected_vals and total > 0:
                # Categorical with declared options
                out_lines.append(f"  {param}:")
                for val in expected_vals:
                    n = counter.get(val, 0)
                    pct = 100 * n / total
                    flag = "  ★ NEVER SELECTED" if n == 0 else ""
                    out_lines.append(f"    {val:<22}  {n:3d}/{total:3d}  ({pct:4.0f}%){flag}")
                # Any unexpected values?
                extras = set(counter) - set(expected_vals)
                for val in extras:
                    n = counter[val]
                    pct = 100 * n / total
                    out_lines.append(f"    {val:<22}  {n:3d}/{total:3d}  ({pct:4.0f}%)  [unexpected]")
            elif total > 0:
                # No declared options — just show counter
                out_lines.append(f"  {param}:")
                for val, n in counter.most_common():
                    pct = 100 * n / total
                    out_lines.append(f"    {val:<22}  {n:3d}/{total:3d}  ({pct:4.0f}%)")
            elif cont_vals:
                # Continuous param
                s = pd.Series(cont_vals)
                out_lines.append(
                    f"  {param}:  min={s.min():.2f}  median={s.median():.2f}  "
                    f"max={s.max():.2f}  n={len(s)}"
                )
                # Histogram (8 bins)
                hist = pd.cut(s, bins=8).value_counts().sort_index()
                for interval, n in hist.items():
                    pct = 100 * n / len(s)
                    bar = "▏" + "█" * int(pct / 2)
                    out_lines.append(f"    {str(interval):<28}  {n:3d}  ({pct:4.0f}%) {bar}")

        # Continuous params not in EXPECTED but observed (e.g., hdh_threshold)
        for param, vals in winners_cont[fe_class].items():
            if param in EXPECTED[fe_class]:
                continue
            if not vals:
                continue
            s = pd.Series(vals)
            out_lines.append(
                f"  {param} (continuous):  min={s.min():.2f}  median={s.median():.2f}  "
                f"max={s.max():.2f}  n={len(s)}"
            )

        # Flag never-selected
        out_lines.append("")
        removal = []
        for param, expected_vals in EXPECTED[fe_class].items():
            counter = winners_cat[fe_class].get(param, Counter())
            for val in expected_vals:
                if counter.get(val, 0) == 0 and sum(counter.values()) > 0:
                    removal.append((param, val))
        if removal:
            out_lines.append(f"  ★ REMOVAL CANDIDATES (0/{n_runs_class} selections):")
            for p, v in removal:
                out_lines.append(f"      {p} = {v}")
        else:
            out_lines.append("  No removal candidates — all options selected at least once.")

    # Model param summary (compact)
    out_lines.append("\n" + "─" * 100)
    out_lines.append("Dataset params (across all FE classes):")
    out_lines.append("─" * 100)
    for k, c in global_ds_cat.items():
        total = sum(c.values())
        out_lines.append(f"  {k}: {dict(c.most_common())}  (n={total})")
    for k, vals in global_ds_cont.items():
        s = pd.Series(vals)
        out_lines.append(
            f"  {k}: min={s.min():.3f}  median={s.median():.3f}  "
            f"max={s.max():.3f}  n={len(s)}"
        )

    text = "\n".join(out_lines)
    print(text)
    out_path = Path("notebooks/phase_a_review/06_search_space_audit.txt")
    out_path.write_text(text + "\n")
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
