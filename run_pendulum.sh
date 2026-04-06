#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# ── Pendulum hyperparameters ──────────────────────────────────────
TRAIN_SEED="${TRAIN_SEED:-42}"
TEST_SEED="${TEST_SEED:-17}"
N_TRAIN="${N_TRAIN:-4000}"
N_TEST="${N_TEST:-1000}"
DT="${DT:-0.05}"
N_ITER="${N_ITER:-60}"
N_RESTARTS="${N_RESTARTS:-2}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"

echo "============================================================"
echo "  Pendulum Experiment Suite"
echo "  train_seed=$TRAIN_SEED  test_seed=$TEST_SEED"
echo "  n_train=$N_TRAIN  n_test=$N_TEST  dt=$DT"
echo "  n_iter=$N_ITER  n_restarts=$N_RESTARTS"
echo "  rollout_steps=$ROLLOUT_STEPS"
echo "============================================================"
echo ""

echo "[1/1] Pendulum: all four method families"
python -m validation.validation_pendulum \
    --train-seed $TRAIN_SEED --test-seed $TEST_SEED \
    --n-train $N_TRAIN --n-test $N_TEST \
    --dt $DT --n-iter $N_ITER --n-restarts $N_RESTARTS \
    --rollout-steps $ROLLOUT_STEPS
echo ""

echo "============================================================"
echo "  Pendulum experiments complete."
echo "  Figures: papers/figures/"
echo "  Data:    papers/data/"
echo "============================================================"
