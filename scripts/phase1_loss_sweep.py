"""Phase 1 of the loss-function & bias study (dev-notes/experiments/
loss_function_and_bias_study.md §5).

Wide-net single-model loss sweep. For each tree family (LGBM, XGB, CatBoost)
we take the production best-MAE member's config and dataset and change ONLY the
training objective, isolating the loss effect within a fixed model class. Two
linear production members (Ridge, Lasso top130) are trained as a fixed reference
so the ensemble diversity / marginal-value analysis mirrors the production
5-family structure.

Everything runs through the repo's canonical split machinery so results are
comparable with production:
  * carve_holdout(HOLDOUT_DAYS=90) before CV,
  * TimeSeriesSplitter(VALIDATION_CV_FOLDS=5, expanding) for the OOF stream,
  * build_pipeline + compute_sample_weights(weight_half_life) exactly as tuning.

Per candidate we collect the full 5-fold OOF prediction stream (leak-free over
the whole pre-holdout pool) plus a fit-on-pool / predict-holdout stream. Because
every candidate shares the same dataset index, splitter and seed, the OOF test
indices align exactly across candidates -> diversity/marginal-value can be
computed directly, no mlruns parquet alignment needed.

Outputs (data/experiments/phase1/):
  * oof_predictions.parquet   wide: index, y_true, one col per candidate (OOF)
  * holdout_predictions.parquet  wide: index, y_true, one col per candidate
  * summary.csv               one row per candidate, the §3 metric panel
  * per_hour.parquet          candidate x hour me/mae/rmse (recent-2y OOF)
  * marginal.csv              diversity + marginal-contribution screen

No production artefact is touched. This is a diagnostic script.
"""

from __future__ import annotations

import json
import time
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

OUT = Path("data/experiments/phase1")
DATASET_DIR = Path("data/processed/datasets")

EVENING_HOURS = [17, 18, 19, 20]
MIDDAY_HOURS = [10, 11, 12, 13, 14]
RECENT_DAYS = 730  # recent-2y reporting slice (training still uses full history)

# ── Per-family fixed base (production best-MAE-selected member) ───────────
# dataset + hyperparams + weight_half_life held constant; only the objective
# changes across candidates within a family.
BASE = {
    "LGBM": {
        "model_type": "LGBMRegressor",
        "dataset": "price_fs_shap_top99",
        "weight_half_life": None,
        "params": {
            "learning_rate": 0.02, "n_estimators": 1200, "max_depth": 10,
            "num_leaves": 1023, "min_child_samples": 50, "subsample": 0.7,
            "subsample_freq": 1, "colsample_bytree": 0.7, "reg_alpha": 3.0,
            "reg_lambda": 5.0, "min_split_gain": 0.1,
        },
    },
    "XGB": {
        "model_type": "XGBRegressor",
        "dataset": "price_fs_shap_top99",
        "weight_half_life": 1095,
        "params": {
            "learning_rate": 0.023, "n_estimators": 1000, "max_depth": 8,
            "min_child_weight": 50, "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 5.0, "reg_lambda": 5.0, "gamma": 0.3,
        },
    },
    "CatBoost": {
        "model_type": "CatBoostRegressor",
        "dataset": "price_fs_shap_top90",
        "weight_half_life": None,
        "params": {
            "learning_rate": 0.018, "iterations": 900, "depth": 6,
            "l2_leaf_reg": 4.0, "subsample": 0.7, "rsm": 0.7,
            "min_child_samples": 50, "verbose": 0,
        },
    },
}

# Fixed linear reference members (production top130 winners), for the ensemble
# diversity/marginal analysis only.
LINEAR_REF = {
    "Ridge": {"model_type": "Ridge", "dataset": "price_fs_shap_top130",
              "params": {"alpha": 10.0}, "scaler": "robust", "weight_half_life": None},
    "Lasso": {"model_type": "Lasso", "dataset": "price_fs_shap_top130",
              "params": {"alpha": 0.03162277660168379}, "scaler": "standard",
              "weight_half_life": None},
}

HUBER_DELTAS = [50, 100, 200]
QUANTILE_TAUS = [0.60, 0.70]
HUBERBIAS_LAMBDAS = [0.05, 0.10, 0.20]
HUBERBIAS_DELTA = 100.0


# ── Custom objectives (sklearn callable) ────────────────────────────────────
# Signature (y_true, y_pred, sample_weight=None) works for BOTH libraries:
#   * LightGBM calls with 2 args (sample_weight=None) and multiplies the
#     returned grad/hess by its own sample weights internally.
#   * XGBoost (>=2) detects the sample_weight param, passes the weights, and
#     does NOT re-apply them — so the callable must weight grad/hess itself.
def _blend_obj(alpha: float):
    """Convex blend  alpha*MAE + (1-alpha)*MSE. MSE core -> converges fast."""
    def obj(y_true, y_pred, sample_weight=None):
        r = y_pred - y_true
        grad = alpha * np.sign(r) + (1.0 - alpha) * r
        hess = np.full_like(r, max(1.0 - alpha, 1e-3))
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess
    return obj


def _huberbias_obj(lam: float, delta: float = HUBERBIAS_DELTA):
    """Huber core (MAE-ish tails, MSE-ish centre) + explicit bias penalty
    lam*(mean residual)^2. The bias term adds a uniform push 2*lam*mean_r to
    every sample's gradient, driving the (weighted) global mean error to zero."""
    def obj(y_true, y_pred, sample_weight=None):
        r = y_pred - y_true
        g = np.clip(r, -delta, delta)
        h = np.where(np.abs(r) <= delta, 1.0, 0.1)
        mean_r = float(np.average(r, weights=sample_weight))
        grad = g + 2.0 * lam * mean_r
        if sample_weight is not None:
            grad = grad * sample_weight
            h = h * sample_weight
        return grad, h
    return obj


def _candidates() -> list[dict]:
    """Build the full candidate list: (id, family, loss params to merge)."""
    cands: list[dict] = []

    def add(family, loss_id, overrides, base_score=None):
        cands.append({"id": f"{family}__{loss_id}", "family": family,
                      "loss_id": loss_id, "overrides": overrides,
                      "base_score": base_score})

    for fam in ("LGBM", "XGB", "CatBoost"):
        if fam == "LGBM":
            add(fam, "mae", {"objective": "l1"})
            add(fam, "rmse", {"objective": "l2"})
            for d in HUBER_DELTAS:
                add(fam, f"huber_d{d}", {"objective": "huber", "alpha": float(d)})
            for t in QUANTILE_TAUS:
                add(fam, f"quantile_t{int(t*100)}",
                    {"objective": "quantile", "alpha": float(t)})
        elif fam == "XGB":
            add(fam, "mae", {"objective": "reg:absoluteerror"})
            add(fam, "rmse", {"objective": "reg:squarederror"})
            for d in HUBER_DELTAS:
                add(fam, f"huber_d{d}",
                    {"objective": "reg:pseudohubererror", "huber_slope": float(d)})
            for t in QUANTILE_TAUS:
                add(fam, f"quantile_t{int(t*100)}",
                    {"objective": "reg:quantileerror", "quantile_alpha": float(t)})
        elif fam == "CatBoost":
            add(fam, "mae", {"loss_function": "MAE"})
            add(fam, "rmse", {"loss_function": "RMSE"})
            for d in HUBER_DELTAS:
                add(fam, f"huber_d{d}", {"loss_function": f"Huber:delta={d}"})
            for t in QUANTILE_TAUS:
                add(fam, f"quantile_t{int(t*100)}",
                    {"loss_function": f"Quantile:alpha={t}"})

    # Custom objectives — LGBM & XGB only (callable objective).
    for fam in ("LGBM", "XGB"):
        mean_y = None  # base_score set at runtime for XGB
        add(fam, "blend_a50", {"objective": _blend_obj(0.5)}, base_score="mean")
        for lam in HUBERBIAS_LAMBDAS:
            add(fam, f"huberbias_l{int(lam*100):02d}",
                {"objective": _huberbias_obj(lam)}, base_score="mean")
    return cands


# ── Training helpers ─────────────────────────────────────────────────────
def _weights_for(index: pd.DatetimeIndex, half_life, ref_start) -> np.ndarray | None:
    if half_life is None:
        return None
    days = (index - ref_start).total_seconds() / 86400.0
    return compute_sample_weights(pd.Series(days), float(half_life))


def _fit_predict_oof(X, y, pool_idx, hold_idx, model_type, params, scaler,
                     half_life, base_score, logtag) -> tuple[pd.Series, pd.Series]:
    """Return (oof_series over pool, holdout_series). Fold indices align across
    candidates because dataset/splitter/seed are shared."""
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    X_hold, y_hold = X.iloc[hold_idx], y.iloc[hold_idx]
    cv = TimeSeriesSplitter(n_splits=VALIDATION_CV_FOLDS, mode="expanding")
    pool_ref = X_pool.index[0]

    def make(params_local):
        p = dict(params_local)
        if base_score == "mean" and model_type == "XGBRegressor":
            p["base_score"] = float(y_pool.mean())
        return _make_model(model_type, p)

    # OOF over the pool. Fold 1 trains on ~2 days (expanding split, oldest
    # 2015-16 data) — custom-objective trees can fail to fit that degenerate
    # fold; leave its OOF unset (those rows are ancient and out of the recent
    # reporting slice / dropped by the marginal screen). Folds 2-5 cover all
    # recent data.
    oof = pd.Series(index=X_pool.index, dtype=float)
    for k, (tr, te) in enumerate(cv.split(X_pool.index)):
        try:
            pipe = build_pipeline(make(params), scaler=scaler, target_transform="none")
            fitp = {}
            w = _weights_for(X_pool.index[tr], half_life, pool_ref)
            if w is not None:
                fitp["model__sample_weight"] = w
            pipe.fit(X_pool.iloc[tr], y_pool.iloc[tr], **fitp)
            oof.iloc[te] = pipe.predict(X_pool.iloc[te])
            logger.info(f"  [{logtag}] fold {k+1}/{VALIDATION_CV_FOLDS} done "
                        f"(train={len(tr)} test={len(te)})")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  [{logtag}] fold {k+1} SKIPPED (train={len(tr)}): {e}")
    oof = oof.dropna()

    # Fit full pool -> predict holdout
    pipe = build_pipeline(make(params), scaler=scaler, target_transform="none")
    fitp = {}
    w = _weights_for(X_pool.index, half_life, pool_ref)
    if w is not None:
        fitp["model__sample_weight"] = w
    pipe.fit(X_pool, y_pool, **fitp)
    hold = pd.Series(pipe.predict(X_hold), index=X_hold.index)
    return oof, hold


# ── Metric panel ─────────────────────────────────────────────────────────
def _block(y_true, y_pred, mask) -> dict:
    m = calculate_metrics(y_true[mask], y_pred[mask])
    return {"mae": m["mae"], "rmse": m["rmse"], "me": m["me"]}


def _panel(oof_true: pd.Series, oof_pred: pd.Series, recent_start) -> dict:
    idx = oof_pred.index
    rec = idx >= recent_start
    yt, yp = oof_true.to_numpy(), oof_pred.to_numpy()
    hours = idx.hour.to_numpy()
    wkday = idx.dayofweek.to_numpy() < 5

    ev = np.isin(hours, EVENING_HOURS)
    md = np.isin(hours, MIDDAY_HOURS)
    row = {}
    for tag, m in (("all", rec), ("evening", rec & ev), ("midday", rec & md)):
        b = _block(yt, yp, m)
        row[f"{tag}_mae"] = b["mae"]
        row[f"{tag}_rmse"] = b["rmse"]
        row[f"{tag}_me"] = b["me"]
    row["evening_me_weekday"] = _block(yt, yp, rec & ev & wkday)["me"]
    row["evening_me_weekend"] = _block(yt, yp, rec & ev & ~wkday)["me"]
    return row


def _per_hour(oof_true: pd.Series, oof_pred: pd.Series, recent_start) -> pd.DataFrame:
    idx = oof_pred.index
    rec = idx >= recent_start
    df = pd.DataFrame({"y": oof_true[rec].to_numpy(), "p": oof_pred[rec].to_numpy(),
                       "hour": idx[rec].hour})
    rows = []
    for h, g in df.groupby("hour"):
        r = g["y"] - g["p"]
        rows.append({"hour": int(h), "me": float(r.mean()),
                     "mae": float(r.abs().mean()),
                     "rmse": float(np.sqrt((r**2).mean()))})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cands = _candidates()
    logger.info(f"Phase 1 loss sweep: {len(cands)} candidates + "
                f"{len(LINEAR_REF)} linear refs")

    # Cache datasets + splits per dataset name (indices align within a dataset).
    ds_cache: dict[str, tuple] = {}

    def get_split(ds_name):
        if ds_name not in ds_cache:
            X, y = load_dataset(str(DATASET_DIR / f"{ds_name}.parquet"))
            pool_idx, hold_idx = carve_holdout(X.index, HOLDOUT_DAYS)
            ds_cache[ds_name] = (X, y, pool_idx, hold_idx)
        return ds_cache[ds_name]

    oof_cols: dict[str, pd.Series] = {}
    hold_cols: dict[str, pd.Series] = {}
    per_hour_frames = []
    summary_rows = []

    # Recent window anchored to the tree datasets' OOF end.
    Xr, _, pr, _ = get_split(BASE["LGBM"]["dataset"])
    recent_start = Xr.index.max().normalize() - pd.Timedelta(days=RECENT_DAYS)
    logger.info(f"recent reporting slice: >= {recent_start}")

    # ---- Candidates
    for i, c in enumerate(cands):
        fam = c["family"]
        base = BASE[fam]
        X, y, pool_idx, hold_idx = get_split(base["dataset"])
        params = {**base["params"], **c["overrides"]}
        tag = c["id"]
        logger.info(f"=== [{i+1}/{len(cands)}] {tag} ===")
        try:
            oof, hold = _fit_predict_oof(
                X, y, pool_idx, hold_idx, base["model_type"], params,
                scaler="none", half_life=base["weight_half_life"],
                base_score=c["base_score"], logtag=tag)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"FAILED {tag}: {e}")
            continue
        oof_true = y.reindex(oof.index)
        oof_cols[tag] = oof
        hold_cols[tag] = hold
        panel = _panel(oof_true, oof, recent_start)
        panel.update({"candidate": tag, "family": fam, "loss": c["loss_id"],
                      "n_oof": int(len(oof))})
        # full-history + holdout overall for context
        panel["holdout_mae"] = calculate_metrics(y.reindex(hold.index), hold)["mae"]
        panel["holdout_rmse"] = calculate_metrics(y.reindex(hold.index), hold)["rmse"]
        panel["holdout_me"] = calculate_metrics(y.reindex(hold.index), hold)["me"]
        summary_rows.append(panel)
        ph = _per_hour(oof_true, oof, recent_start)
        ph["candidate"] = tag
        per_hour_frames.append(ph)
        # Crash-tolerant checkpoint: persist this candidate immediately and
        # rewrite the running summary (cheap at ~32 rows).
        (OUT / "oof").mkdir(exist_ok=True)
        (OUT / "hold").mkdir(exist_ok=True)
        oof.to_frame("pred").to_parquet(OUT / "oof" / f"{tag}.parquet")
        hold.to_frame("pred").to_parquet(OUT / "hold" / f"{tag}.parquet")
        pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
        pd.concat(per_hour_frames, ignore_index=True).to_parquet(OUT / "per_hour.parquet")
        logger.info(f"  {tag}: recent all mae={panel['all_mae']:.2f} "
                    f"me={panel['all_me']:.2f} | evening me={panel['evening_me']:.2f} "
                    f"mae={panel['evening_mae']:.2f} | midday me={panel['midday_me']:.2f}")

    # ---- Linear reference members (for ensemble analysis)
    for name, spec in LINEAR_REF.items():
        X, y, pool_idx, hold_idx = get_split(spec["dataset"])
        logger.info(f"=== linear ref {name} ===")
        try:
            oof, hold = _fit_predict_oof(
                X, y, pool_idx, hold_idx, spec["model_type"], spec["params"],
                scaler=spec["scaler"], half_life=spec["weight_half_life"],
                base_score=None, logtag=name)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"FAILED linear {name}: {e}")
            continue
        oof_cols[name] = oof
        hold_cols[name] = hold

    # ---- Assemble wide prediction frames (align on shared tree index)
    tree_ds = BASE["LGBM"]["dataset"]
    X, y, pool_idx, hold_idx = get_split(tree_ds)
    oof_index = y.iloc[pool_idx].index
    hold_index = y.iloc[hold_idx].index

    def wide(cols: dict[str, pd.Series], index, truth) -> pd.DataFrame:
        df = pd.DataFrame(index=index)
        df["y_true"] = truth.reindex(index)
        for k, s in cols.items():
            df[k] = s.reindex(index)
        return df

    oof_wide = wide(oof_cols, oof_index, y)
    hold_wide = wide(hold_cols, hold_index, y)
    oof_wide.to_parquet(OUT / "oof_predictions.parquet")
    hold_wide.to_parquet(OUT / "holdout_predictions.parquet")
    pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
    if per_hour_frames:
        pd.concat(per_hour_frames, ignore_index=True).to_parquet(OUT / "per_hour.parquet")

    # ---- Diversity + marginal-value screen (OOF, recent-2y for reporting;
    # weights fit inverse-MAE on the last 90 OOF days = production convention).
    _marginal_screen(oof_wide, recent_start)

    logger.info(f"Phase 1 DONE in {(time.time()-t0)/60:.1f} min. Outputs in {OUT}")


def _marginal_screen(oof_wide: pd.DataFrame, recent_start) -> None:
    """Incumbent = MAE tree members + Ridge + Lasso, inverse-MAE weighted on the
    last-90-day OOF tail. For each candidate: error-correlation with the
    incumbent ensemble (overall + evening, recent-2y) and marginal contribution
    when added (holdout-tail mae/rmse/me delta, production optimistic convention).
    """
    df = oof_wide.dropna(how="any").copy()
    idx = df.index
    y = df["y_true"].to_numpy()
    tail_start = idx.max().normalize() - pd.Timedelta(days=HOLDOUT_DAYS)
    tail = idx >= tail_start
    rec = idx >= recent_start
    ev = np.isin(idx.hour, EVENING_HOURS)

    incumbent = [c for c in ["LGBM__mae", "XGB__mae", "CatBoost__mae",
                             "Ridge", "Lasso"] if c in df.columns]

    def inv_mae_weights(members):
        maes = np.array([np.mean(np.abs(y[tail] - df[m].to_numpy()[tail]))
                         for m in members])
        w = 1.0 / maes
        return w / w.sum()

    def ens_pred(members, w):
        return sum(wi * df[m].to_numpy() for wi, m in zip(w, members))

    w0 = inv_mae_weights(incumbent)
    p0 = ens_pred(incumbent, w0)
    base_tail = {k: v for k, v in
                 zip(("mae", "rmse", "me"),
                     (np.mean(np.abs(y[tail]-p0[tail])),
                      np.sqrt(np.mean((y[tail]-p0[tail])**2)),
                      np.mean(y[tail]-p0[tail])))}
    incumbent_err_rec = (y - p0)[rec]

    rows = []
    for cand in df.columns:
        if cand in ("y_true",) or cand in incumbent:
            continue
        cand_err_rec = (y - df[cand].to_numpy())[rec]
        corr_all = float(np.corrcoef(cand_err_rec, incumbent_err_rec)[0, 1])
        evr = ev[rec]
        corr_ev = float(np.corrcoef(cand_err_rec[evr], incumbent_err_rec[evr])[0, 1])
        members = incumbent + [cand]
        w1 = inv_mae_weights(members)
        p1 = ens_pred(members, w1)
        rows.append({
            "candidate": cand,
            "err_corr_all": corr_all,
            "err_corr_evening": corr_ev,
            "cand_weight": float(w1[-1]),
            "d_mae": float(np.mean(np.abs(y[tail]-p1[tail])) - base_tail["mae"]),
            "d_rmse": float(np.sqrt(np.mean((y[tail]-p1[tail])**2)) - base_tail["rmse"]),
            "d_me": float(np.mean(y[tail]-p1[tail]) - base_tail["me"]),
        })
    out = pd.DataFrame(rows).sort_values("d_mae")
    out.to_csv(OUT / "marginal.csv", index=False)
    with open(OUT / "incumbent.json", "w") as f:
        json.dump({"members": incumbent, "weights": w0.tolist(),
                   "tail_start": str(tail_start), "base_tail": base_tail}, f, indent=2)
    logger.info(f"marginal screen: incumbent tail mae={base_tail['mae']:.3f} "
                f"me={base_tail['me']:.3f}; {len(rows)} candidates scored")


if __name__ == "__main__":
    main()
