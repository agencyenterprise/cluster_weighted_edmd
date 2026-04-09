#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Standard seeds for statistical validation
DEFAULT_SEEDS=(1 42 101 307 1001 7789 13245 11 103 13)

if [ $# -eq 0 ]; then
    SEEDS=("${DEFAULT_SEEDS[@]}")
else
    SEEDS=("$@")
fi

SEEDS_STR="${SEEDS[*]}"
N_SEEDS=${#SEEDS[@]}

echo "============================================================"
echo "  Residual-Aware Bayesian Clustering — Full Experiment Suite"
echo "  Statistical seeds (${N_SEEDS}): ${SEEDS_STR}"
echo "============================================================"
echo ""

# ── Statistical validation (multi-seed) ───────────────────────────
echo "[1/2] Lorenz: statistical validation (${N_SEEDS} seeds, CIs, p-values)"
python -m validation.validation_lorenz_statistical --seeds ${SEEDS_STR}
echo ""

echo "[2/2] Pendulum: statistical validation (${N_SEEDS} seeds, CIs, p-values)"
python -m validation.validation_pendulum_statistical --seeds ${SEEDS_STR}
echo ""

echo "============================================================"
echo "  All experiments complete."
echo "  Figures: papers/figures/"
echo "  Data:    papers/data/"
echo "============================================================"
