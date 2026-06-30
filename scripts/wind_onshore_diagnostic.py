"""Investigate the persistent ~14% wind_onshore gap vs EMA on the backtest window.

We know:
- v4 BACKTEST MAE ratio is 1.14 (was 1.28 in v2). EMA still beats us by ~260 MWh at DE_NATIONAL.
- All 4 wind_onshore TSO ensembles are summed at DE_NATIONAL.
- EMA cached forecasts only have national-level columns, so per-region direct
  comparison vs EMA is not possible. We can still:
  * Decompose our DE_NATIONAL error into per-region contributions
  * Plot/aggregate residuals per month and per hour-of-day to see when the gap shows up
  * Examine the residual error AT the DE_NATIONAL level vs EMA on the same window
  * Check whether per-region MAE is dominated by one TSO (likely DE_TENNET)

Outputs (notebooks/phase_a_review/07_wind_onshore_deep_dive.txt):
1. Per-region BACKTEST MAE on apples-to-apples (forecast weather both sides)
2. Monthly MAE breakdown DE_NATIONAL: ours vs EMA, with delta
3. Conditional analysis at DE_NATIONAL on the backtest:
   - by hour-of-day (peak vs off-peak)
   - by weekday/weekend
   - by actual_load bucket (high-wind hours vs low-wind hours)
4. Year-over-year improvement check: 2022/2023/2024 (hindcast — EMA uses actuals)
   vs 2025+ (backtest — EMA uses forecasts) gap pattern
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHED_BACKTEST = Path(
    "/home/smnfrs/projects/energy_prices/data/ema/historical/backtest_2025.parquet"
)
CACHED_HINDCAST = Path(
    "/home/smnfrs/projects/energy_prices/data/ema/historical/hindcast_2022_01_to_2024_12.parquet"
)
HF_DIR = Path("data/processed/historical_forecasts")
WIND_ONSHORE_REGIONS = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNETBW"]
EMA_COL = "prognostizierte_erzeugung_onshore"


def _load_ema() -> pd.DataFrame:
    bt = pd.read_parquet(CACHED_BACKTEST)
    hc = pd.read_parquet(CACHED_HINDCAST)
    bt["__src__"] = "backtest"
    hc["__src__"] = "hindcast"
    df = pd.concat([bt, hc], axis=0).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _load_region(region: str) -> pd.DataFrame:
    p = HF_DIR / f"wind_onshore_{region}.parquet"
    df = pd.read_parquet(p)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _load_national() -> pd.DataFrame:
    return _load_region("DE_NATIONAL")


def _mae(s: pd.Series) -> float:
    return float(s.abs().mean())


def _rmse(s: pd.Series) -> float:
    return float(np.sqrt((s**2).mean()))


def per_region_breakdown(out: list[str]) -> pd.DataFrame:
    """Per-region MAE on backtest window. Plus the MAE *contribution* to
    national error (correlated, not just sum-of-MAEs)."""
    out.append("\n" + "=" * 100)
    out.append("Section 1: Per-region wind_onshore MAE on BACKTEST window (2025-01-08+)")
    out.append("=" * 100)

    nat = _load_national()
    nat["err"] = nat["y_pred"] - nat["y_true"]

    regions_data = {}
    for r in WIND_ONSHORE_REGIONS:
        df = _load_region(r)
        df["err"] = df["y_pred"] - df["y_true"]
        regions_data[r] = df

    bt_start = pd.Timestamp("2025-01-08", tz="UTC")
    bt_end = pd.Timestamp("2026-03-06 23:00", tz="UTC")

    rows = []
    for r, df in regions_data.items():
        sub = df.loc[bt_start:bt_end].dropna(subset=["err"])
        nat_sub = nat.loc[bt_start:bt_end].dropna(subset=["err"])
        # Joint frame: per-region err and national err on same hours
        joint = pd.concat({
            "reg_err": sub["err"],
            "nat_err": nat_sub["err"],
        }, axis=1).dropna()

        rows.append({
            "region": r,
            "n": len(sub),
            "mean_actual": float(sub["y_true"].mean()),
            "max_actual": float(sub["y_true"].max()),
            "mae": _mae(sub["err"]),
            "rmse": _rmse(sub["err"]),
            "bias": float(sub["err"].mean()),
            "corr_with_nat_err": float(joint.corr().iloc[0, 1]) if len(joint) > 1 else float("nan"),
        })

    df = pd.DataFrame(rows)
    out.append(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # Aggregate sanity: sum of region forecasts vs national forecast
    nat_sub = nat.loc[bt_start:bt_end]
    sum_pred = sum(
        regions_data[r].loc[bt_start:bt_end]["y_pred"].reindex(nat_sub.index)
        for r in WIND_ONSHORE_REGIONS
    )
    sum_true = sum(
        regions_data[r].loc[bt_start:bt_end]["y_true"].reindex(nat_sub.index)
        for r in WIND_ONSHORE_REGIONS
    )
    out.append("")
    out.append("Aggregate cross-check (national should equal sum-of-regions):")
    diff_pred = (nat_sub["y_pred"] - sum_pred).abs().mean()
    diff_true = (nat_sub["y_true"] - sum_true).abs().mean()
    out.append(f"  |nat_pred - sum(region_preds)| mean={diff_pred:.2f}")
    out.append(f"  |nat_true - sum(region_trues)| mean={diff_true:.2f}")
    return df


def monthly_breakdown(out: list[str]) -> pd.DataFrame:
    """DE_NATIONAL monthly MAE: ours vs EMA on backtest window."""
    out.append("\n" + "=" * 100)
    out.append("Section 2: Monthly DE_NATIONAL wind_onshore MAE — ours vs EMA")
    out.append("=" * 100)

    nat = _load_national()
    ema = _load_ema()

    bt_start = pd.Timestamp("2025-01-08", tz="UTC")
    bt_end = pd.Timestamp("2026-03-06 23:00", tz="UTC")

    df = pd.DataFrame({
        "y_true": nat["y_true"],
        "y_pred": nat["y_pred"],
        "ema_pred": ema[EMA_COL],
        "src": ema["__src__"],
    }).dropna()

    df["ours_err"] = df["y_pred"] - df["y_true"]
    df["ema_err"] = df["ema_pred"] - df["y_true"]
    df["month"] = df.index.to_period("M")

    bt_df = df.loc[bt_start:bt_end]
    rows = []
    for m, g in bt_df.groupby("month"):
        rows.append({
            "month": str(m),
            "n": len(g),
            "ema_mae": _mae(g["ema_err"]),
            "ours_mae": _mae(g["ours_err"]),
            "ema_rmse": _rmse(g["ema_err"]),
            "ours_rmse": _rmse(g["ours_err"]),
            "delta_mae": _mae(g["ours_err"]) - _mae(g["ema_err"]),
            "ratio": _mae(g["ours_err"]) / _mae(g["ema_err"]),
            "mean_actual": float(g["y_true"].mean()),
        })
    out_df = pd.DataFrame(rows)
    out.append(out_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    return out_df


def conditional_breakdown(out: list[str]) -> None:
    """Conditional MAE analysis on the backtest window."""
    out.append("\n" + "=" * 100)
    out.append("Section 3: Conditional MAE — DE_NATIONAL backtest")
    out.append("=" * 100)

    nat = _load_national()
    ema = _load_ema()

    bt_start = pd.Timestamp("2025-01-08", tz="UTC")
    bt_end = pd.Timestamp("2026-03-06 23:00", tz="UTC")

    df = pd.DataFrame({
        "y_true": nat["y_true"],
        "y_pred": nat["y_pred"],
        "ema_pred": ema[EMA_COL],
        "src": ema["__src__"],
    }).dropna().loc[bt_start:bt_end]

    df["ours_err"] = df["y_pred"] - df["y_true"]
    df["ema_err"] = df["ema_pred"] - df["y_true"]

    # By hour-of-day
    df_local = df.copy()
    df_local["hour"] = df_local.index.tz_convert("Europe/Berlin").hour
    g = df_local.groupby("hour").agg(
        n=("ours_err", "size"),
        ema_mae=("ema_err", lambda s: s.abs().mean()),
        ours_mae=("ours_err", lambda s: s.abs().mean()),
    )
    g["delta"] = g["ours_mae"] - g["ema_mae"]
    g["ratio"] = g["ours_mae"] / g["ema_mae"]
    out.append("\nBy hour-of-day (Europe/Berlin):")
    out.append(g.to_string(float_format=lambda x: f"{x:.1f}"))

    # By weekday/weekend
    df_local["dow"] = df_local.index.tz_convert("Europe/Berlin").dayofweek
    df_local["is_weekend"] = df_local["dow"] >= 5
    g = df_local.groupby("is_weekend").agg(
        n=("ours_err", "size"),
        ema_mae=("ema_err", lambda s: s.abs().mean()),
        ours_mae=("ours_err", lambda s: s.abs().mean()),
    )
    g["delta"] = g["ours_mae"] - g["ema_mae"]
    g["ratio"] = g["ours_mae"] / g["ema_mae"]
    out.append("\nBy weekend vs weekday:")
    out.append(g.to_string(float_format=lambda x: f"{x:.1f}"))

    # By wind generation level (proxy for wind speed)
    df["actual_bucket"] = pd.qcut(df["y_true"], 5, labels=["q1_low", "q2", "q3", "q4", "q5_high"])
    g = df.groupby("actual_bucket", observed=True).agg(
        n=("ours_err", "size"),
        mean_actual=("y_true", "mean"),
        ema_mae=("ema_err", lambda s: s.abs().mean()),
        ours_mae=("ours_err", lambda s: s.abs().mean()),
    )
    g["delta"] = g["ours_mae"] - g["ema_mae"]
    g["ratio"] = g["ours_mae"] / g["ema_mae"]
    out.append("\nBy wind-output quintile:")
    out.append(g.to_string(float_format=lambda x: f"{x:.1f}"))

    # Bias check
    out.append("\nBias (positive = over-forecast):")
    out.append(f"  EMA bias:  {df['ema_err'].mean():+.1f}")
    out.append(f"  ours bias: {df['ours_err'].mean():+.1f}")

    # Error-level coupling — when EMA is off, are we also off in the same direction?
    corr = df[["ema_err", "ours_err"]].corr().iloc[0, 1]
    out.append(f"\nResidual correlation (ours_err, ema_err): {corr:.3f}")
    # Hours where ours is materially worse
    diff = df["ours_err"].abs() - df["ema_err"].abs()
    worse = diff[diff > 500]
    out.append(f"\nHours where |ours_err| - |ema_err| > 500 MWh: {len(worse)} / {len(df)} "
               f"({100*len(worse)/len(df):.1f}%)")
    out.append(f"  these hours contribute extra MAE = {diff[diff > 0].sum() / len(df):.1f} MWh/hr")


def yoy_breakdown(out: list[str]) -> None:
    """Year-over-year MAE pattern across hindcast (2022-24) + backtest (2025+)."""
    out.append("\n" + "=" * 100)
    out.append("Section 4: Year-over-year MAE — hindcast vs backtest")
    out.append("=" * 100)
    out.append("(NOTE: hindcast EMA uses actual weather; backtest EMA uses forecast weather.")
    out.append(" Our model uses forecast weather across the entire range.)")

    nat = _load_national()
    ema = _load_ema()

    df = pd.DataFrame({
        "y_true": nat["y_true"],
        "y_pred": nat["y_pred"],
        "ema_pred": ema[EMA_COL],
        "src": ema["__src__"],
    }).dropna()

    df["ours_err"] = df["y_pred"] - df["y_true"]
    df["ema_err"] = df["ema_pred"] - df["y_true"]
    df["year"] = df.index.year

    rows = []
    for (year, src), g in df.groupby(["year", "src"]):
        rows.append({
            "year": year,
            "src": src,
            "n": len(g),
            "ema_mae": _mae(g["ema_err"]),
            "ours_mae": _mae(g["ours_err"]),
            "ratio": _mae(g["ours_err"]) / _mae(g["ema_err"]),
            "mean_actual": float(g["y_true"].mean()),
        })
    out_df = pd.DataFrame(rows)
    out.append(out_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


def main():
    out: list[str] = []
    out.append("=" * 100)
    out.append("wind_onshore deep dive — investigating the v4 ~14% gap vs EMA on backtest")
    out.append("=" * 100)

    per_region_breakdown(out)
    monthly_breakdown(out)
    conditional_breakdown(out)
    yoy_breakdown(out)

    text = "\n".join(out)
    print(text)
    out_path = Path("notebooks/phase_a_review/07_wind_onshore_deep_dive.txt")
    out_path.write_text(text + "\n")
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
