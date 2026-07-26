"""Phase 2 of the loss-function & bias study (dev-notes/experiments/
loss_function_and_bias_study.md §6, Tracks 1 & 2).

NO RETRAINING. Pure post-processing of the leak-free prediction columns Phase 1
saved to data/experiments/phase1/ (oof_predictions.parquet,
holdout_predictions.parquet). Two tracks:

  Track 1 — Loss-diverse ensembles (H2). Fix the weighting (production
    inverse-MAE) and vary the MEMBER SET: incumbent all-MAE bloc vs incumbent +
    bias-aware members. Does a "worse-alone" bias-aware model earn a seat by
    flattening the blended diurnal bias?

  Track 2 — Weighting-strategy sweep (the crux). Fix a diverse member set and
    vary the WEIGHTING: inverse-MAE (production), inverse-RMSE, MAE-optimal
    (SLSQP simplex), bias-penalised (minimise MAE + gamma*|evening ME|),
    category-floored SLSQP. Compare the evening-ME vs MAE vs RMSE frontier. This
    is the §1 "weighting trap": inverse-MAE rewards the biased median members and
    fights the bias-aware ones.

Evaluation surfaces (all on the recent regime):
  * WALK-FORWARD (honest, primary): fit weights on the last 90 OOF days
    (2026-01-22 -> 2026-04-22), evaluate on the disjoint 90-day holdout
    (2026-04-23 -> 2026-07-21) that sits immediately after. This is a genuine
    forward split — no in-sample leakage in the eval numbers.
  * IN-SAMPLE reference (production convention): fit AND evaluate on the holdout,
    to show the optimism gap vs the honest walk-forward.
  * OOF recent-2y (>= 2024-07-21): the long, stable per-hour / evening diurnal
    diagnostic (weights carried over from the walk-forward fit).

ME sign: repo convention ME = actual - forecast, so +ve ME = UNDER-prediction
(opposite to the dashboard's forecast-actual sign). Evening = 17-20 UTC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

P1 = Path("data/experiments/phase1")
OUT = Path("data/experiments/phase2")

EVENING_HOURS = [17, 18, 19, 20]
MIDDAY_HOURS = [10, 11, 12, 13, 14]
RECENT_DAYS = 730
WEIGHT_FIT_DAYS = 90  # production fits inverse-MAE weights on the recent 90 days

INCUMBENT = ["LGBM__mae", "XGB__mae", "CatBoost__mae", "Ridge", "Lasso"]

# Bias-aware members carried forward from Phase 1 §5a (best MAE-vs-bias frontier
# + most-complementary in the marginal screen).
BIAS_AWARE = ["LGBM__blend_a50", "LGBM__huberbias_l20", "XGB__huberbias_l10",
              "CatBoost__rmse"]

CATEGORY_OF = {  # member -> family, for the category floor
    "LGBM__mae": "lgbm", "LGBM__rmse": "lgbm", "LGBM__blend_a50": "lgbm",
    "LGBM__huberbias_l10": "lgbm", "LGBM__huberbias_l20": "lgbm",
    "XGB__mae": "xgboost", "XGB__rmse": "xgboost", "XGB__huberbias_l10": "xgboost",
    "CatBoost__mae": "catboost", "CatBoost__rmse": "catboost",
    "Ridge": "linear", "Lasso": "linear",
}


# ── Metric panel ─────────────────────────────────────────────────────────
def _panel(y: np.ndarray, p: np.ndarray, hours: np.ndarray) -> dict:
    ev = np.isin(hours, EVENING_HOURS)
    md = np.isin(hours, MIDDAY_HOURS)
    r = y - p  # ME = actual - forecast; +ve = under-predict

    def blk(m):
        rr = r[m]
        return (float(np.mean(np.abs(rr))),
                float(np.sqrt(np.mean(rr ** 2))),
                float(np.mean(rr)))

    mae, rmse, me = blk(np.ones_like(y, bool))
    ev_mae, ev_rmse, ev_me = blk(ev)
    md_mae, md_rmse, md_me = blk(md)
    return {"mae": mae, "rmse": rmse, "me": me,
            "ev_me": ev_me, "ev_mae": ev_mae, "ev_rmse": ev_rmse,
            "md_me": md_me, "md_mae": md_mae}


def _ens(df: pd.DataFrame, members, w) -> np.ndarray:
    return sum(wi * df[m].to_numpy() for wi, m in zip(w, members))


# ── Weighting strategies (all return a simplex weight vector) ─────────────
def w_inverse_mae(dff, members, y):
    maes = np.array([np.mean(np.abs(y - dff[m].to_numpy())) for m in members])
    w = 1.0 / maes
    return w / w.sum()


def w_inverse_rmse(dff, members, y):
    rmses = np.array([np.sqrt(np.mean((y - dff[m].to_numpy()) ** 2)) for m in members])
    w = 1.0 / rmses
    return w / w.sum()


def w_simple(dff, members, y):
    return np.full(len(members), 1.0 / len(members))


def _slsqp(members, P, y, loss, floor=0.0):
    """Minimise `loss(resid, hours)` over the simplex, w >= floor."""
    n = len(members)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(floor, 1.0)] * n
    x0 = np.full(n, 1.0 / n)
    res = minimize(lambda w: loss(y - P @ w), x0, method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = np.clip(res.x, floor, None)
    return w / w.sum()


def w_mae_optimal(dff, members, y, floor=0.0):
    P = np.column_stack([dff[m].to_numpy() for m in members])
    return _slsqp(members, P, y, lambda r: np.mean(np.abs(r)), floor)


def w_mae_floored(dff, members, y):
    return w_mae_optimal(dff, members, y, floor=0.02)


def make_bias_penalised(hours, gamma):
    def w_fn(dff, members, y):
        P = np.column_stack([dff[m].to_numpy() for m in members])
        ev = np.isin(hours, EVENING_HOURS)

        def loss(r):
            return np.mean(np.abs(r)) + gamma * abs(np.mean(r[ev]))
        return _slsqp(members, P, y, loss, floor=0.0)
    return w_fn


# ── Evaluation harness ───────────────────────────────────────────────────
def evaluate(name, members, weight_fn, fit_df, fit_hours,
             eval_frames) -> dict:
    """Fit weights on fit_df, evaluate the resulting panel on each eval frame."""
    yf = fit_df["y_true"].to_numpy()
    w = weight_fn(fit_df, members, yf)
    row = {"config": name, "n_members": len(members)}
    row["weights"] = {m: round(float(wi), 3) for m, wi in zip(members, w)}
    for tag, (edf, ehours) in eval_frames.items():
        p = _ens(edf, members, w)
        pan = _panel(edf["y_true"].to_numpy(), p, ehours)
        for k, v in pan.items():
            row[f"{tag}_{k}"] = v
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(P1 / "oof_predictions.parquet")
    hold = pd.read_parquet(P1 / "holdout_predictions.parquet")

    # Recent-2y slice anchored to the FULL dataset end (= holdout end), matching
    # Phase 1's recent_start.
    recent_start = hold.index.max().normalize() - pd.Timedelta(days=RECENT_DAYS)
    # Weight-fit window = last 90 OOF days (immediately precedes the holdout).
    fit_start = oof.index.max().normalize() - pd.Timedelta(days=WEIGHT_FIT_DAYS)

    all_members = sorted(set(INCUMBENT + BIAS_AWARE))
    oof_fit = oof.loc[oof.index >= fit_start, ["y_true"] + all_members].dropna(how="any")
    oof_rec = oof.loc[oof.index >= recent_start, ["y_true"] + all_members].dropna(how="any")
    hold_all = hold[["y_true"] + all_members].dropna(how="any")

    fit_hours = oof_fit.index.hour.to_numpy()
    eval_frames = {
        "hold": (hold_all, hold_all.index.hour.to_numpy()),      # walk-forward eval
        "oofrec": (oof_rec, oof_rec.index.hour.to_numpy()),      # long diurnal diag
    }
    # In-sample reference: fit AND eval on holdout (production optimism).
    eval_frames_insample = {"hold": (hold_all, hold_all.index.hour.to_numpy())}

    print(f"OOF fit window:  {oof_fit.index.min()} -> {oof_fit.index.max()} "
          f"({len(oof_fit)} rows)")
    print(f"Holdout eval:    {hold_all.index.min()} -> {hold_all.index.max()} "
          f"({len(hold_all)} rows)")
    print(f"OOF recent-2y:   {oof_rec.index.min()} -> {oof_rec.index.max()} "
          f"({len(oof_rec)} rows)\n")

    rows = []

    # ── Track 1: member-set sweep at fixed inverse-MAE weighting ──────────
    print("=" * 78)
    print("TRACK 1 — member-set sweep (weighting fixed = inverse-MAE, "
          "walk-forward fit on OOF tail)")
    print("=" * 78)
    member_sets = {
        "T1_incumbent(5,MAE)": INCUMBENT,
        "T1_+LGBM_blend": INCUMBENT + ["LGBM__blend_a50"],
        "T1_+LGBM_hbias20": INCUMBENT + ["LGBM__huberbias_l20"],
        "T1_+XGB_hbias10": INCUMBENT + ["XGB__huberbias_l10"],
        "T1_+CatBoost_rmse": INCUMBENT + ["CatBoost__rmse"],
        "T1_+all4_biasaware": INCUMBENT + BIAS_AWARE,
        "T1_biasaware_swap": ["LGBM__blend_a50", "XGB__huberbias_l10",
                              "CatBoost__rmse", "Ridge", "Lasso"],
    }
    for name, members in member_sets.items():
        f = oof_fit[["y_true"] + members].dropna(how="any")
        row = evaluate(name, members, w_inverse_mae, f, f.index.hour.to_numpy(),
                       eval_frames)
        rows.append({**row, "track": 1})

    _print_table([r for r in rows if r["track"] == 1])

    # ── Track 2: weighting-strategy sweep on a fixed diverse member set ───
    print("\n" + "=" * 78)
    print("TRACK 2 — weighting sweep (member set fixed = incumbent + 4 bias-aware)")
    print("=" * 78)
    diverse = INCUMBENT + BIAS_AWARE
    fd = oof_fit[["y_true"] + diverse].dropna(how="any")
    fh = fd.index.hour.to_numpy()
    hd = hold[["y_true"] + diverse].dropna(how="any")
    ef = {"hold": (hd, hd.index.hour.to_numpy()),
          "oofrec": (oof_rec[["y_true"] + diverse].dropna(how="any"),
                     oof_rec[["y_true"] + diverse].dropna(how="any").index.hour.to_numpy())}

    weight_strategies = {
        "W_inverse_mae(prod)": w_inverse_mae,
        "W_inverse_rmse": w_inverse_rmse,
        "W_simple_avg": w_simple,
        "W_mae_optimal_slsqp": w_mae_optimal,
        "W_mae_floored_2pct": w_mae_floored,
        "W_biaspen_g0.5": make_bias_penalised(fh, 0.5),
        "W_biaspen_g1.0": make_bias_penalised(fh, 1.0),
        "W_biaspen_g2.0": make_bias_penalised(fh, 2.0),
    }
    t2 = []
    for name, wfn in weight_strategies.items():
        row = evaluate(name, diverse, wfn, fd, fh, ef)
        t2.append({**row, "track": 2})
    rows.extend(t2)
    _print_table(t2)

    # In-sample reference for the two anchor strategies (production optimism gap).
    print("\n--- in-sample reference (fit AND eval on holdout) ---")
    ref = []
    for name, wfn in {"W_inverse_mae(prod)": w_inverse_mae,
                      "W_mae_optimal_slsqp": w_mae_optimal}.items():
        # fit on holdout itself
        hh = hd.index.hour.to_numpy()
        row = evaluate(name + "[insample]", diverse, wfn, hd, hh,
                       {"hold": (hd, hh)})
        ref.append(row)
    _print_table(ref)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "results.csv", index=False)
    # weights detail
    with open(OUT / "weights.txt", "w") as fp:
        for r in rows:
            fp.write(f"{r['config']}: {r['weights']}\n")
    print(f"\nSaved -> {OUT}/results.csv, {OUT}/weights.txt")


def _print_table(rows) -> None:
    hdr = (f"{'config':26s} {'n':>2s} | "
           f"{'HOLD mae':>8s} {'rmse':>6s} {'me':>6s} {'ev_me':>6s} {'ev_mae':>7s} "
           f"{'md_me':>6s} | {'OOFrec ev_me':>12s} {'mae':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        oof_ev = r.get("oofrec_ev_me", float("nan"))
        oof_mae = r.get("oofrec_mae", float("nan"))
        print(f"{r['config']:26s} {r['n_members']:>2d} | "
              f"{r['hold_mae']:8.2f} {r['hold_rmse']:6.2f} {r['hold_me']:6.2f} "
              f"{r['hold_ev_me']:6.2f} {r['hold_ev_mae']:7.2f} {r['hold_md_me']:6.2f} | "
              f"{oof_ev:12.2f} {oof_mae:6.2f}")


if __name__ == "__main__":
    main()
