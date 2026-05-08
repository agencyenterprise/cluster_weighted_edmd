#!/usr/bin/env bash
#
# Multi-seed, multi-config statistical validation for the damped pendulum.
# Iterates every YAML config in config/pendulum/ (or a custom set passed
# as positional args) through validation/run_statistical.py --config.
#
# Each config writes one JSON + one PNG into papers/data/ / papers/figures/
# under the config's `name:` field, so multiple configs do NOT overwrite.
#
# Usage
# -----
#   ./run_pendulum.sh                                  # all configs
#   ./run_pendulum.sh config/pendulum/uniform_baseline.yaml
#   ./run_pendulum.sh config/pendulum/dt_*.yaml        # glob

set -e
cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
    CONFIGS=(config/pendulum/*.yaml)
else
    CONFIGS=("$@")
fi

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

MAX_WORKERS="${MAX_WORKERS:-0}"
N_ITER_CAP="${N_ITER_CAP:-}"
N_RESTARTS_CAP="${N_RESTARTS_CAP:-}"
N_TRAIN_CAP="${N_TRAIN_CAP:-}"
N_TEST_CAP="${N_TEST_CAP:-}"
ROLLOUT_STEPS_CAP="${ROLLOUT_STEPS_CAP:-}"
EXTRA_FLAGS=(--max-workers "$MAX_WORKERS" --run-id "$RUN_ID")
[ -n "$N_ITER_CAP" ]         && EXTRA_FLAGS+=(--n-iter-cap         "$N_ITER_CAP")
[ -n "$N_RESTARTS_CAP" ]     && EXTRA_FLAGS+=(--n-restarts-cap     "$N_RESTARTS_CAP")
[ -n "$N_TRAIN_CAP" ]        && EXTRA_FLAGS+=(--n-train-cap        "$N_TRAIN_CAP")
[ -n "$N_TEST_CAP" ]         && EXTRA_FLAGS+=(--n-test-cap         "$N_TEST_CAP")
[ -n "$ROLLOUT_STEPS_CAP" ]  && EXTRA_FLAGS+=(--rollout-steps-cap  "$ROLLOUT_STEPS_CAP")
N_CFG=${#CONFIGS[@]}

echo "============================================================"
echo "  Pendulum Statistical Validation Suite"
echo "  Run ID:        ${RUN_ID}"
echo "  Output dir:    papers/data/${RUN_ID}/   papers/figures/${RUN_ID}/"
echo "  ${N_CFG} config(s) to run:"
for cfg in "${CONFIGS[@]}"; do
    echo "    - ${cfg}"
done
echo "============================================================"

idx=0
for cfg in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    echo ""
    echo "############################################################"
    echo "##  [${idx}/${N_CFG}] ${cfg}"
    echo "############################################################"
    python -m validation.run_statistical --config "$cfg" "${EXTRA_FLAGS[@]}"
    echo ""
done

echo "============================================================"
echo "  All ${N_CFG} Pendulum config(s) complete (Run ID: ${RUN_ID})."
echo "  Data:    papers/data/${RUN_ID}/<config_name>.json"
echo "  Figures: papers/figures/${RUN_ID}/<config_name>.png"
echo "============================================================"
