#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# ── Lorenz hyperparameters ────────────────────────────────────────
SEED="${SEED:-42}"
N_STEPS="${N_STEPS:-5000}"
DT="${DT:-0.01}"
WARMUP="${WARMUP:-1000}"
N_TRAIN="${N_TRAIN:-4000}"
N="${N:-5}"
N_ITER="${N_ITER:-100}"
N_RESTARTS="${N_RESTARTS:-3}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-500}"
MS_RANGE="${MS_RANGE:-2 13}"
MS_RESTARTS="${MS_RESTARTS:-2}"
SWEEP_N="${SWEEP_N:-3 5 8 12 20 30 50}"
HYBRID_N="${HYBRID_N:-5 12 20 30 50}"

COMMON="--seed $SEED --n-steps $N_STEPS --dt $DT --warmup $WARMUP"

echo "============================================================"
echo "  Lorenz Experiment Suite"
echo "  seed=$SEED  n_steps=$N_STEPS  dt=$DT  warmup=$WARMUP"
echo "  n_train=$N_TRAIN  N=$N  n_iter=$N_ITER  n_restarts=$N_RESTARTS"
echo "============================================================"
echo ""

echo "[1/5] Full experiment (sanity + fit + model selection)"
python -m validation.validation_lorenz \
    $COMMON --N $N --n-iter $N_ITER --n-restarts $N_RESTARTS \
    --ms-range $MS_RANGE --ms-restarts $MS_RESTARTS
echo ""

echo "[2/5] One-step + rollout vs global EDMD"
python -m validation.validation_lorenz_vs_edmd \
    $COMMON --n-train $N_TRAIN --N $N --n-iter $N_ITER \
    --n-restarts $N_RESTARTS --rollout-steps $ROLLOUT_STEPS
echo ""

echo "[3/5] Sweep N (all methods)"
python -m validation.validation_lorenz_sweep_N \
    $COMMON --n-train $N_TRAIN --n-iter $N_ITER \
    --n-restarts $MS_RESTARTS --N-values $SWEEP_N
echo ""

echo "[4/5] Taylor-analytic vs Taylor-LS vs GMM"
python -m validation.validation_lorenz_hybrid \
    $COMMON --n-train $N_TRAIN --n-iter $N_ITER \
    --n-restarts $MS_RESTARTS --N-values $HYBRID_N
echo ""

echo "[5/5] Local EDMD vs global EDMD"
python -m validation.validation_lorenz_local_edmd \
    $COMMON --n-train $N_TRAIN --n-iter $N_ITER \
    --n-restarts $MS_RESTARTS
echo ""

echo "============================================================"
echo "  Lorenz experiments complete."
echo "  Figures: papers/figures/"
echo "  Data:    papers/data/"
echo "============================================================"
