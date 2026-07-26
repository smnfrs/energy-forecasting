"""Phase 2 / Track 3 of the loss-function & bias study (dev-notes/experiments/
loss_function_and_bias_study.md §6, Track 3 — distribution shaping, H4).

RETRAINS a small grid. Tests whether modelling a variance-stabilised / skew-
reduced TARGET (log_shift, yeo_johnson) reduces the diurnal bias at the root,
independently of the loss lever. To isolate the transform effect we hold the
objective at MAE (production baseline) and vary only `target_transform`, for LGBM
and XGB (CatBoost dropped — weakest family, §5a finding 5).

Everything runs through the exact repo split used in Phase 1 (reuses BASE,
_weights_for, _panel from scripts/phase1_loss_sweep). The `none` rows are the
Phase 1 MAE baselines, recomputed here so the comparison is self-contained.

ME = actual - forecast (+ve = under-predict). Evening = 17-20 UTC. Recent-2y OOF
slice + 90-day holdout, same as Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from energy_forecasting.config.modeling import HOLDOUT_DAYS, VALIDATION_CV_FOLDS
from energy_forecasting.modeling.cv import TimeSeriesSplitter, carve_holdout
from energy_forecasting.modeling.datasets import load_dataset
from energy_forecasting.modeling.metrics import calculate_metrics
from energy_forecasting.modeling.tuning import _make_model
from energy_forecasting.modeling.training import build_pipeline, compute_sample_weights

from phase1_loss_sweep import BASE, RECENT_DAYS, _panel

OUT = Path("data/experiments/phase3")
DATASET_DIR = Path("data/processed/datasets")
TRANSFORMS = ["none", "log_shift", "yeo_johnson"]
FAMILIES = ["LGBM", "XGB"]
NATIVE_MAE = {"LGBM": {"objective": "l1"}, "XGB": {"objective": "reg:absoluteerror"}}


def _weights_for(index, half_life, ref_start):
    if half_life is None:
        return None
    days = (index - ref_start).total_seconds() / 86400.0
    return compute_sample_weights(pd.Series(days), float(half_life))


def _fit_predict(X, y, pool_idx, hold_idx, model_type, params, half_life,
                 target_transform, logtag):
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    X_hold = X.iloc[hold_idx]
    cv = TimeSeriesSplitter(n_splits=VALIDATION_CV_FOLDS, mode="expanding")
    pool_ref = X_pool.index[0]

    oof = pd.Series(index=X_pool.index, dtype=float)
    for k, (tr, te) in enumerate(cv.split(X_pool.index)):
        try:
            pipe = build_pipeline(_make_model(model_type, params), scaler="none",
                                  target_transform=target_transform)
            fitp = {}
            w = _weights_for(X_pool.index[tr], half_life, pool_ref)
            if w is not None:
                fitp["model__sample_weight"] = w
            pipe.fit(X_pool.iloc[tr], y_pool.iloc[tr], **fitp)
            oof.iloc[te] = pipe.predict(X_pool.iloc[te])
            logger.info(f"  [{logtag}] fold {k+1}/{VALIDATION_CV_FOLDS} "
                        f"(train={len(tr)} test={len(te)})")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  [{logtag}] fold {k+1} SKIPPED (train={len(tr)}): {e}")
    oof = oof.dropna()

    pipe = build_pipeline(_make_model(model_type, params), scaler="none",
                          target_transform=target_transform)
    fitp = {}
    w = _weights_for(X_pool.index, half_life, pool_ref)
    if w is not None:
        fitp["model__sample_weight"] = w
    pipe.fit(X_pool, y_pool, **fitp)
    hold = pd.Series(pipe.predict(X_hold), index=X_hold.index)
    return oof, hold


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds_cache = {}

    def get_split(ds):
        if ds not in ds_cache:
            X, y = load_dataset(str(DATASET_DIR / f"{ds}.parquet"))
            pool_idx, hold_idx = carve_holdout(X.index, HOLDOUT_DAYS)
            ds_cache[ds] = (X, y, pool_idx, hold_idx)
        return ds_cache[ds]

    Xr, _, _, _ = get_split(BASE["LGBM"]["dataset"])
    recent_start = Xr.index.max().normalize() - pd.Timedelta(days=RECENT_DAYS)

    rows = []
    for fam in FAMILIES:
        base = BASE[fam]
        X, y, pool_idx, hold_idx = get_split(base["dataset"])
        for tf in TRANSFORMS:
            tag = f"{fam}__mae__{tf}"
            logger.info(f"=== {tag} ===")
            params = {**base["params"], **NATIVE_MAE[fam]}
            oof, hold = _fit_predict(X, y, pool_idx, hold_idx, base["model_type"],
                                     params, base["weight_half_life"], tf, tag)
            oof_true = y.reindex(oof.index)
            panel = _panel(oof_true, oof, recent_start)
            hm = calculate_metrics(y.reindex(hold.index), hold)
            # holdout evening/midday
            hi = hold.index
            ev = np.isin(hi.hour, [17, 18, 19, 20])
            yr, pr = y.reindex(hi).to_numpy(), hold.to_numpy()
            ev_me = float(np.mean((yr - pr)[ev]))
            ev_mae = float(np.mean(np.abs((yr - pr)[ev])))
            rows.append({
                "candidate": tag, "family": fam, "transform": tf,
                "oof_all_mae": panel["all_mae"], "oof_all_me": panel["all_me"],
                "oof_ev_me": panel["evening_me"], "oof_ev_mae": panel["evening_mae"],
                "oof_md_me": panel["midday_me"],
                "hold_mae": hm["mae"], "hold_rmse": hm["rmse"], "hold_me": hm["me"],
                "hold_ev_me": ev_me, "hold_ev_mae": ev_mae,
            })
            oof.to_frame("pred").to_parquet(OUT / f"oof_{tag}.parquet")
            hold.to_frame("pred").to_parquet(OUT / f"hold_{tag}.parquet")
            pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
            logger.info(f"  {tag}: OOF mae={panel['all_mae']:.2f} ev_me="
                        f"{panel['evening_me']:.2f} | HOLD mae={hm['mae']:.2f} "
                        f"ev_me={ev_me:.2f}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("TRACK 3 — target transform (loss fixed = MAE). ME = actual-forecast, "
          "+ve = under-predict")
    print("=" * 92)
    print(f"{'candidate':22s} | {'OOF mae':>7s} {'ev_me':>6s} {'ev_mae':>6s} {'md_me':>6s}"
          f" | {'HOLD mae':>8s} {'rmse':>6s} {'ev_me':>6s} {'ev_mae':>6s}")
    print("-" * 92)
    for _, r in df.iterrows():
        print(f"{r['candidate']:22s} | {r['oof_all_mae']:7.2f} {r['oof_ev_me']:6.2f} "
              f"{r['oof_ev_mae']:6.2f} {r['oof_md_me']:6.2f} | {r['hold_mae']:8.2f} "
              f"{r['hold_rmse']:6.2f} {r['hold_ev_me']:6.2f} {r['hold_ev_mae']:6.2f}")
    print(f"\nSaved -> {OUT}/results.csv")


if __name__ == "__main__":
    main()
