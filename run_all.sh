#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
    echo "Usage: ./run_all.sh SEED [SEED ...]"
    echo "Example: ./run_all.sh 42 43 44 45 46"
    exit 1
fi

SEEDS=("$@")
SEEDS_STR="${SEEDS[*]}"
N_SEEDS=${#SEEDS[@]}

# ── Shared hyperparameters ────────────────────────────────────────
# Override any of these via environment variables before running.
# Lorenz
export SEED="${SEED:-42}"
export N_STEPS="${N_STEPS:-5000}"
export WARMUP="${WARMUP:-1000}"
export N="${N:-5}"
export N_ITER="${N_ITER:-100}"
export N_RESTARTS="${N_RESTARTS:-3}"
export ROLLOUT_STEPS="${ROLLOUT_STEPS:-500}"
# Pendulum
export TRAIN_SEED="${TRAIN_SEED:-42}"
export TEST_SEED="${TEST_SEED:-17}"
export N_TRAIN="${N_TRAIN:-4000}"
export N_TEST="${N_TEST:-1000}"

echo "============================================================"
echo "  Residual-Aware Bayesian Clustering — Full Experiment Suite"
echo "  Statistical seeds (${N_SEEDS}): ${SEEDS_STR}"
echo "============================================================"
echo ""

# ── Single-seed experiments ───────────────────────────────────────
echo "=== Lorenz (single seed) ==="
./run_lorenz.sh
echo ""

echo "=== Pendulum (single seed) ==="
./run_pendulum.sh
echo ""

# ── Statistical validation (multi-seed) ───────────────────────────
echo "[7/8] Lorenz: statistical validation (${N_SEEDS} seeds, CIs, p-values)"
python -m validation.validation_lorenz_statistical --seeds ${SEEDS_STR}
echo ""

echo "[8/8] Pendulum: statistical validation (${N_SEEDS} seeds, CIs, p-values)"
python -m validation.validation_pendulum_statistical --seeds ${SEEDS_STR}
echo ""

echo "============================================================"
echo "  All experiments complete."
echo "  Figures: papers/figures/"
echo "  Data:    papers/data/"
echo "============================================================"
