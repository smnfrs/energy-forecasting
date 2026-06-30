"""Phase A v4 review — compare v4 (post 2026-05-06 18:18 CEST) vs v2 vs EMA.

v4 is the rerun with EMA-aligned (expanded) FE search spaces and the physics
NaN fix. v2 is the prior production run that produced the historical forecasts
at notebooks/phase_a_review/03_full_range_vs_upstream.txt.

Outputs three sections:
1. Per-(target,region,model_type) base-run cv_mae: v4 vs v2 + delta
2. Per-(target,region) ensemble holdout MAE: v4 vs v2
3. Backtest-window MAE/RMSE for DE_NATIONAL ensembles: v4 vs v2 vs EMA cached
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from energy_forecasting.config.modeling import EXPERIMENTS

# v4 cutoff: training was launched 2026-05-06 18:18 CEST = 16:18 UTC.
# v2 cutoff: prior run was launched 2026-05-04 (Stage 5b production).
V4_START_UTC = pd.Timestamp("2026-05-06 16:00:00+0000")
V4_END_UTC = pd.Timestamp("2026-05-07 03:00:00+0000")
V2_END_UTC = pd.Timestamp("2026-05-06 16:00:00+0000")

GEN_TARGETS = ["wind_onshore", "wind_offshore", "solar", "load", "gen_load_diff"]
TARGET_TO_EXP = {
    "wind_onshore": "gen_wind_onshore",
    "wind_offshore": "gen_wind_offshore",
    "solar": "gen_solar",
    "load": "gen_load",
    "gen_load_diff": "gen_gen_load_diff",
}
MODEL_TYPES = ["LGBMRegressor", "XGBRegressor", "ElasticNet"]

CACHED_BACKTEST = Path(
    "/home/smnfrs/projects/energy_prices/data/ema/historical/backtest_2025.parquet"
)
CACHED_HINDCAST = Path(
    "/home/smnfrs/projects/energy_prices/data/ema/historical/hindcast_2022_01_to_2024_12.parquet"
)
HF_DIR = Path("data/processed/historical_forecasts")

# EMA column → our (target, region) keys at DE_NATIONAL level
EMA_COL_MAP = {
    "wind_onshore": "prognostizierte_erzeugung_onshore",
    "wind_offshore": "prognostizierte_erzeugung_offshore",
    "solar": "prognostizierte_erzeugung_photovoltaik",
    "load": "prognostizierter_verbrauch_gesamt",
    "gen_load_diff": "_gen_load_diff",
}


def _runs_in_window(
    client: MlflowClient,
    experiment_path: str,
    target: str,
    region: str,
    feature_version: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> list:
    exp = client.get_experiment_by_name(experiment_path)
    if exp is None:
        return []
    filt = (
        f"tags.target='{target}' AND tags.region='{region}' "
        f"AND tags.feature_version='{feature_version}' "
        f"AND attributes.status='FINISHED'"
    )
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=filt,
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    out = []
    for r in runs:
        st_ms = r.info.start_time
        st_utc = pd.Timestamp(st_ms, unit="ms", tz="UTC")
        if start_utc <= st_utc <= end_utc:
            out.append((st_utc, r))
    return out


def collect_run_metrics() -> pd.DataFrame:
    client = MlflowClient()
    rows: list[dict] = []

    for target in GEN_TARGETS:
        from energy_forecasting.config.modeling import GEN_LOAD_TARGETS

        regions = GEN_LOAD_TARGETS[target]["regions"]
        exp_path = EXPERIMENTS[TARGET_TO_EXP[target]]

        for region in regions:
            for model_type in MODEL_TYPES:
                feature_version = f"optuna_{model_type.lower()}"

                v4_runs = _runs_in_window(
                    client, exp_path, target, region, feature_version,
                    V4_START_UTC, V4_END_UTC,
                )
                v2_runs = _runs_in_window(
                    client, exp_path, target, region, feature_version,
                    pd.Timestamp("2026-05-01", tz="UTC"), V2_END_UTC,
                )

                v4 = v4_runs[0] if v4_runs else None
                v2 = v2_runs[0] if v2_runs else None

                def _m(r, key):
                    if r is None:
                        return None
                    return r[1].data.metrics.get(key)

                rows.append({
                    "target": target,
                    "region": region,
                    "model_type": model_type,
                    "v4_run": v4[1].info.run_id[:8] if v4 else None,
                    "v4_cv_mae": _m(v4, "cv_mae"),
                    "v4_holdout_mae": _m(v4, "holdout_mae"),
                    "v2_run": v2[1].info.run_id[:8] if v2 else None,
                    "v2_cv_mae": _m(v2, "cv_mae"),
                    "v2_holdout_mae": _m(v2, "holdout_mae"),
                })
    df = pd.DataFrame(rows)
    df["delta_cv_mae"] = df["v4_cv_mae"] - df["v2_cv_mae"]
    df["delta_pct"] = 100.0 * df["delta_cv_mae"] / df["v2_cv_mae"]
    return df


def collect_ensemble_metrics() -> pd.DataFrame:
    client = MlflowClient()
    rows = []
    for target in GEN_TARGETS:
        from energy_forecasting.config.modeling import GEN_LOAD_TARGETS

        regions = GEN_LOAD_TARGETS[target]["regions"]
        exp_path = EXPERIMENTS[TARGET_TO_EXP[target]]
        for region in regions:
            v4 = _runs_in_window(
                client, exp_path, target, region, "ensemble",
                V4_START_UTC, V4_END_UTC,
            )
            v2 = _runs_in_window(
                client, exp_path, target, region, "ensemble",
                pd.Timestamp("2026-05-01", tz="UTC"), V2_END_UTC,
            )
            def _m(r, key):
                if not r:
                    return None
                return r[0][1].data.metrics.get(key)

            rows.append({
                "target": target,
                "region": region,
                "v4_run": v4[0][1].info.run_id[:8] if v4 else None,
                "v4_mae": _m(v4, "mae"),
                "v4_rmse": _m(v4, "rmse"),
                "v4_pi_coverage": _m(v4, "pi_coverage"),
                "v2_run": v2[0][1].info.run_id[:8] if v2 else None,
                "v2_mae": _m(v2, "mae"),
                "v2_rmse": _m(v2, "rmse"),
                "v2_pi_coverage": _m(v2, "pi_coverage"),
            })
    return pd.DataFrame(rows)


def _load_ema_cached() -> pd.DataFrame:
    bt = pd.read_parquet(CACHED_BACKTEST)
    hc = pd.read_parquet(CACHED_HINDCAST)
    bt["__source__"] = "backtest"
    hc["__source__"] = "hindcast"
    df = pd.concat([bt, hc], axis=0).sort_index()
    # Drop overlapping hindcast rows that overlap with backtest
    df = df[~df.index.duplicated(keep="first")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    # Compute EMA's gen_load_diff = generation_total - load_total
    # (matches our gen_load_diff = sum(all generation) - load convention).
    if "prognostizierte_erzeugung_gesamt" in df.columns and "prognostizierter_verbrauch_gesamt" in df.columns:
        df["_gen_load_diff"] = (
            df["prognostizierte_erzeugung_gesamt"]
            - df["prognostizierter_verbrauch_gesamt"]
        )
    return df


def backtest_compare() -> pd.DataFrame:
    """For each DE_NATIONAL target, compare ensemble forecast vs EMA cached
    forecasts on the apples-to-apples backtest window (2025-01-08+).
    """
    ema = _load_ema_cached()
    rows = []

    for target, ema_col in EMA_COL_MAP.items():
        hf_path = HF_DIR / f"{target}_DE_NATIONAL.parquet"
        if not hf_path.exists():
            continue
        ours = pd.read_parquet(hf_path)
        # Expected: index = timestamp, columns include "actual" and "forecast"
        if ours.index.tz is None:
            ours.index = ours.index.tz_localize("UTC")

        # Find prediction column
        pred_col = None
        actual_col = None
        for cand in ["y_pred", "forecast", "prediction", "ensemble_pred"]:
            if cand in ours.columns:
                pred_col = cand
                break
        for cand in ["y_true", "actual", "y_actual"]:
            if cand in ours.columns:
                actual_col = cand
                break
        if pred_col is None or actual_col is None:
            print(f"  WARN: {hf_path.name} columns: {list(ours.columns)} — skipping")
            continue

        # Find EMA prediction column (suffix-tolerant)
        ema_match = [c for c in ema.columns if c.startswith(ema_col)]
        if not ema_match:
            print(f"  WARN: no EMA col matching '{ema_col}' — skipping")
            continue

        # Build paired frame: ours pred + EMA pred + actual
        df = pd.DataFrame({
            "ours_pred": ours[pred_col],
            "actual": ours[actual_col],
            "ema_pred": ema[ema_match[0]],
            "src": ema["__source__"],
        }).dropna()

        if df.empty:
            continue

        # BACKTEST window only: where EMA used issued forecast (apples-to-apples)
        bt_df = df[df["src"] == "backtest"]
        full_df = df

        for label, sub in [("BACKTEST", bt_df), ("FULL", full_df)]:
            if sub.empty:
                continue
            ours_err = sub["ours_pred"] - sub["actual"]
            ema_err = sub["ema_pred"] - sub["actual"]
            rows.append({
                "target": target,
                "window": label,
                "n": len(sub),
                "start": sub.index.min(),
                "end": sub.index.max(),
                "ema_mae": float(ema_err.abs().mean()),
                "ema_rmse": float(np.sqrt((ema_err**2).mean())),
                "ours_mae": float(ours_err.abs().mean()),
                "ours_rmse": float(np.sqrt((ours_err**2).mean())),
            })
    df = pd.DataFrame(rows)
    df["mae_ratio"] = df["ours_mae"] / df["ema_mae"]
    df["rmse_ratio"] = df["ours_rmse"] / df["ema_rmse"]
    return df


def main():
    print("=" * 100)
    print("Phase A v4 review")
    print("=" * 100)

    print("\n--- Section 1: Per (target, region, model) base-run cv_mae — v4 vs v2 ---\n")
    base = collect_run_metrics()
    base_disp = base.copy()
    for c in ["v4_cv_mae", "v2_cv_mae", "delta_cv_mae", "v4_holdout_mae", "v2_holdout_mae"]:
        base_disp[c] = base_disp[c].apply(lambda x: f"{x:8.1f}" if pd.notna(x) else "    n/a ")
    base_disp["delta_pct"] = base_disp["delta_pct"].apply(lambda x: f"{x:+6.1f}%" if pd.notna(x) else "  n/a ")
    print(base_disp[["target", "region", "model_type",
                     "v4_cv_mae", "v2_cv_mae", "delta_cv_mae", "delta_pct"]].to_string(index=False))

    # Aggregate: % of (target,region,model) combos that improved
    base_clean = base.dropna(subset=["v4_cv_mae", "v2_cv_mae"])
    n_better = (base_clean["delta_cv_mae"] < 0).sum()
    n_total = len(base_clean)
    print(f"\n  v4 better cv_mae in {n_better}/{n_total} combos "
          f"({100*n_better/n_total:.0f}%); "
          f"mean Δ = {base_clean['delta_cv_mae'].mean():+.1f} "
          f"({base_clean['delta_pct'].mean():+.1f}%)")

    # Per-target aggregate
    print("\n  Per-target aggregates:")
    agg = base_clean.groupby("target").agg(
        n=("delta_cv_mae", "size"),
        n_better=("delta_cv_mae", lambda s: (s < 0).sum()),
        mean_delta=("delta_cv_mae", "mean"),
        mean_pct=("delta_pct", "mean"),
    )
    print(agg.to_string())

    print("\n--- Section 2: Ensemble holdout MAE/RMSE — v4 vs v2 ---\n")
    ens = collect_ensemble_metrics()
    ens["delta_mae"] = ens["v4_mae"] - ens["v2_mae"]
    ens["delta_pct"] = 100.0 * ens["delta_mae"] / ens["v2_mae"]
    ens_disp = ens.copy()
    for c in ["v4_mae", "v2_mae", "v4_rmse", "v2_rmse"]:
        ens_disp[c] = ens_disp[c].apply(lambda x: f"{x:8.1f}" if pd.notna(x) else "    n/a ")
    ens_disp["delta_mae"] = ens_disp["delta_mae"].apply(lambda x: f"{x:+7.1f}" if pd.notna(x) else "   n/a ")
    ens_disp["delta_pct"] = ens_disp["delta_pct"].apply(lambda x: f"{x:+6.1f}%" if pd.notna(x) else "  n/a ")
    print(ens_disp[["target", "region", "v4_mae", "v2_mae", "delta_mae",
                    "delta_pct", "v4_rmse", "v2_rmse"]].to_string(index=False))

    ens_clean = ens.dropna(subset=["v4_mae", "v2_mae"])
    if not ens_clean.empty:
        n_better = (ens_clean["delta_mae"] < 0).sum()
        print(f"\n  Ensemble v4 better in {n_better}/{len(ens_clean)} (target,region) combos; "
              f"mean ΔMAE = {ens_clean['delta_mae'].mean():+.1f} ({ens_clean['delta_pct'].mean():+.1f}%)")

    print("\n--- Section 3: DE_NATIONAL ensemble vs EMA (full historical window) ---\n")
    bt = backtest_compare()
    if bt.empty:
        print("  (no historical_forecasts files found)")
    else:
        for _, row in bt.iterrows():
            print(f"  {row['target']:14s} {row['window']:8s} "
                  f"n={row['n']:>6d}  span {row['start'].date()} → {row['end'].date()}  "
                  f"EMA: MAE={row['ema_mae']:6.0f} RMSE={row['ema_rmse']:6.0f}  "
                  f"ours: MAE={row['ours_mae']:6.0f} RMSE={row['ours_rmse']:6.0f}  "
                  f"ratio: {row['mae_ratio']:.2f}/{row['rmse_ratio']:.2f}")

        # Aggregate
        print("\n  Aggregate over BACKTEST window (apples-to-apples, both forecast weather):")
        bt_only = bt[bt["window"] == "BACKTEST"]
        if not bt_only.empty:
            print(f"    Mean MAE  — EMA: {bt_only['ema_mae'].mean():.1f}  "
                  f"ours: {bt_only['ours_mae'].mean():.1f}  "
                  f"ratio: {bt_only['ours_mae'].mean()/bt_only['ema_mae'].mean():.3f}")
            print(f"    Mean RMSE — EMA: {bt_only['ema_rmse'].mean():.1f}  "
                  f"ours: {bt_only['ours_rmse'].mean():.1f}  "
                  f"ratio: {bt_only['ours_rmse'].mean()/bt_only['ema_rmse'].mean():.3f}")
            print(f"    ours beats EMA on MAE in {(bt_only['mae_ratio'] < 1).sum()}/{len(bt_only)}")
            print(f"    ours beats EMA on RMSE in {(bt_only['rmse_ratio'] < 1).sum()}/{len(bt_only)}")

        # v2 baseline (from notebooks/phase_a_review/03_full_range_vs_upstream.txt)
        # for delta tracking over the same backtest window.
        v2_baseline = {
            "wind_onshore":  {"mae": 2389.7, "rmse": 3150.6},
            "wind_offshore": {"mae":  536.9, "rmse":  726.6},
            "solar":         {"mae":  914.1, "rmse": 1769.0},
            "load":          {"mae": 1114.9, "rmse": 1575.5},
            "gen_load_diff": {"mae": 1816.2, "rmse": 2298.6},
        }
        print("\n  Per-target v4 vs v2 BACKTEST delta (negative = v4 better):")
        for _, row in bt_only.iterrows():
            v2 = v2_baseline.get(row["target"])
            if v2 is None:
                continue
            d_mae = row["ours_mae"] - v2["mae"]
            d_rmse = row["ours_rmse"] - v2["rmse"]
            print(f"    {row['target']:14s}  v4 MAE={row['ours_mae']:7.1f}  v2 MAE={v2['mae']:7.1f}  "
                  f"Δ={d_mae:+7.1f} ({100*d_mae/v2['mae']:+5.1f}%)   "
                  f"v4 RMSE={row['ours_rmse']:7.1f}  v2 RMSE={v2['rmse']:7.1f}  "
                  f"Δ={d_rmse:+7.1f} ({100*d_rmse/v2['rmse']:+5.1f}%)")

    return base, ens, bt


if __name__ == "__main__":
    main()
