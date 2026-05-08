#!/usr/bin/env bash
#
# Multi-seed statistical validation of the residual-aware clustering
# pipeline on the unforced double-well Duffing oscillator.
#
# Usage
# -----
#   ./run_duffing.sh                           # full 10-seed sweep, defaults
#   ./run_duffing.sh 1 42 101                  # custom seed list
#   N_ITER=100 N_RESTARTS=3 ./run_duffing.sh   # override fit params
#   SAMPLING=trajectory ./run_duffing.sh       # trajectory-ensemble training data
#
# Outputs
# -------
#   papers/data/duffing_statistical.json   raw per-seed metrics
#   papers/figures/duffing_statistical.png error-bar comparison plot

set -e
cd "$(dirname "$0")"

# -- Standard seeds (match run_all.sh / Lorenz / pendulum statistical) --------

DEFAULT_SEEDS=(1 42 101 307 1001 7789 13245 11 103 13)

if [ $# -eq 0 ]; then
    SEEDS=("${DEFAULT_SEEDS[@]}")
else
    SEEDS=("$@")
fi
SEEDS_STR="${SEEDS[*]}"
N_SEEDS=${#SEEDS[@]}

# -- Hyperparameters (override via env vars) ----------------------------------

# Data sampling
SAMPLING="${SAMPLING:-uniform}"           # uniform | trajectory
N_TRAIN="${N_TRAIN:-4000}"
N_TEST="${N_TEST:-1000}"
BOX_X="${BOX_X:-2.0}"
BOX_XDOT="${BOX_XDOT:-2.0}"
N_TRAJ="${N_TRAJ:-200}"
TRAJ_STEPS="${TRAJ_STEPS:-50}"
IC_X="${IC_X:-2.0}"
IC_XDOT="${IC_XDOT:-2.5}"

# EM fitting
N_ITER="${N_ITER:-80}"
N_RESTARTS="${N_RESTARTS:-2}"

# Rollout
DT="${DT:-0.05}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-400}"

# Method sweep (space-separated lists)
N_LIST="${N_LIST:-2 4 8 16}"
EDMD_DEGS="${EDMD_DEGS:-2 3 4 5}"
LE2_N_LIST="${LE2_N_LIST:-2 4 8 16}"
LE3_N_LIST="${LE3_N_LIST:-2 4 8}"

# -- Banner -------------------------------------------------------------------

echo "============================================================"
echo "  Duffing Statistical Validation Suite"
echo "  seeds (${N_SEEDS}): ${SEEDS_STR}"
echo "  sampling=$SAMPLING"
if [ "$SAMPLING" = "uniform" ]; then
    echo "    n_train=$N_TRAIN  n_test=$N_TEST"
    echo "    box=[-$BOX_X,$BOX_X] x [-$BOX_XDOT,$BOX_XDOT]"
else
    echo "    n_traj=$N_TRAJ  traj_steps=$TRAJ_STEPS"
    echo "    IC box=[-$IC_X,$IC_X] x [-$IC_XDOT,$IC_XDOT]"
fi
echo "  fit: n_iter=$N_ITER  n_restarts=$N_RESTARTS"
echo "  rollout: dt=$DT  rollout_steps=$ROLLOUT_STEPS"
echo "  N_LIST=[$N_LIST]  EDMD_DEGS=[$EDMD_DEGS]"
echo "  LE2_N_LIST=[$LE2_N_LIST]  LE3_N_LIST=[$LE3_N_LIST]"
echo "============================================================"
echo ""

# -- Run ----------------------------------------------------------------------

python -m validation.validation_duffing_statistical \
    --seeds        ${SEEDS_STR} \
    --sampling     "$SAMPLING" \
    --n-train      "$N_TRAIN" \
    --n-test       "$N_TEST" \
    --box-x        "$BOX_X" \
    --box-xdot     "$BOX_XDOT" \
    --n-traj       "$N_TRAJ" \
    --traj-steps   "$TRAJ_STEPS" \
    --ic-x         "$IC_X" \
    --ic-xdot      "$IC_XDOT" \
    --n-iter       "$N_ITER" \
    --n-restarts   "$N_RESTARTS" \
    --dt           "$DT" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --N-list       ${N_LIST} \
    --edmd-degs    ${EDMD_DEGS} \
    --le2-N-list   ${LE2_N_LIST} \
    --le3-N-list   ${LE3_N_LIST}

echo ""
echo "============================================================"
echo "  Duffing statistical validation complete."
echo "  Figures: papers/figures/duffing_statistical.png"
echo "  Data:    papers/data/duffing_statistical.json"
echo "============================================================"
