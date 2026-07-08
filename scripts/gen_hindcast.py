#!/usr/bin/env python3
"""Standalone: export last 7 days of national gen/load predictions to deploy/data/gen_load_hindcast.json.

Reads y_pred from data/processed/historical_forecasts/*_DE_NATIONAL.parquet
(accumulated OOF + production predictions) so the dashboard can overlay past
model predictions against SMARD actuals.  Called by deploy_static.yml after
restoring the deploy-state cache.  No energy_forecasting imports needed.
"""
import json
import pathlib
import sys

HF_DIR = pathlib.Path("data/processed/historical_forecasts")
OUT = pathlib.Path("deploy/data/gen_load_hindcast.json")

if not HF_DIR.exists():
    print(f"No historical_forecasts dir at {HF_DIR}, skipping", file=sys.stderr)
    sys.exit(0)

try:
    import pandas as pd
except ImportError:
    print("pandas not available, skipping hindcast", file=sys.stderr)
    sys.exit(0)

TARGETS = ["wind_onshore", "wind_offshore", "solar", "load", "gen_load_diff"]
result: dict = {}

for target in TARGETS:
    path = HF_DIR / f"{target}_DE_NATIONAL.parquet"
    if not path.exists():
        print(f"  {target}: {path} not found, skipping", file=sys.stderr)
        continue
    try:
        hf = pd.read_parquet(path)[["y_pred"]]
    except Exception as exc:
        print(f"  {target}: read error — {exc}", file=sys.stderr)
        continue

    idx = hf.index
    if hasattr(idx, "tz") and idx.tz is not None:
        hf.index = idx.tz_convert("UTC")
    else:
        hf.index = idx.tz_localize("UTC")

    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=7)
    hf = hf[(hf.index > cutoff) & (hf.index < now)].dropna()
    if hf.empty:
        print(f"  {target}: no data in last 7 days, skipping", file=sys.stderr)
        continue

    result[target] = [
        {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "forecast": round(float(v), 1),
        }
        for ts, v in hf["y_pred"].items()
    ]
    print(f"  {target}: {len(result[target])} entries "
          f"({result[target][0]['timestamp']} → {result[target][-1]['timestamp']})")

if result:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"Written {OUT} ({len(result)} targets)")
else:
    print("No hindcast data — gen_load_hindcast.json not written", file=sys.stderr)
