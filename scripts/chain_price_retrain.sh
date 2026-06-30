#!/usr/bin/env bash
# Chain a fresh price-training run once the in-flight run (PID arg) exits.
#
# Used after the 2026-05-28 overnight finishes: deletes the stale dataset
# caches (price_*.parquet) and Optuna SQLite studies so the new SMARD/EMA
# waterfall + trimmed search spaces take effect, then launches the new
# pipeline under setsid+nohup so it survives shell disconnect.

set -euo pipefail

WAIT_PID="${1:-977002}"
REPO=/home/smnfrs/projects/energy-forecasting
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/logs/train_price_chained_${STAMP}.log"

cd "$REPO"

echo "[chain] waiting for PID $WAIT_PID to exit ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done
echo "[chain] PID $WAIT_PID exited at $(date -Is)"

# Brief grace so MLflow / Optuna finalise on-disk writes.
sleep 10

# Cleanup stale caches — the EMA overlay + trimmed grids require rebuilds.
echo "[chain] removing stale price_*.parquet caches"
rm -fv data/processed/datasets/price_max.parquet \
       data/processed/datasets/price_slim.parquet \
       data/processed/datasets/price_full.parquet \
       data/processed/datasets/price_fs_*.parquet 2>/dev/null || true

echo "[chain] removing stale Optuna studies"
rm -fv data/optuna/fs_shap_top*.db data/optuna/fs_rfecv_*.db 2>/dev/null || true

# Launch the new run. --top-k 4, --use-rfecv, full FS sweep.
echo "[chain] starting new training run, log: $LOG"
/home/smnfrs/miniconda3/envs/energy-forecasting/bin/energy-forecasting \
    train price \
        --feature-selection \
        --top-k 4 \
        --use-rfecv \
    >"$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "${LOG}.pid"
echo "[chain] launched PID $NEW_PID"
wait "$NEW_PID"
echo "[chain] training exited at $(date -Is) with status $?"
