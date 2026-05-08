#!/usr/bin/env bash
#
# Multi-seed, multi-config statistical validation for the Lorenz-63 system.
# Iterates every YAML config in config/lorenz/ (or a custom set passed
# as positional args) through validation/run_statistical.py --config.
#
# Each config writes one JSON + one PNG into papers/data/ / papers/figures/
# under the config's `name:` field, so multiple configs do NOT overwrite.
#
# Usage
# -----
#   ./run_lorenz.sh                                  # all configs
#   ./run_lorenz.sh config/lorenz/attractor_baseline.yaml
#   ./run_lorenz.sh config/lorenz/dt_*.yaml          # glob

set -e
cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
    CONFIGS=(config/lorenz/*.yaml)
else
    CONFIGS=("$@")
fi

MAX_WORKERS="${MAX_WORKERS:-0}"
N_ITER_CAP="${N_ITER_CAP:-}"
N_RESTARTS_CAP="${N_RESTARTS_CAP:-}"
N_TRAIN_CAP="${N_TRAIN_CAP:-}"
N_TEST_CAP="${N_TEST_CAP:-}"
ROLLOUT_STEPS_CAP="${ROLLOUT_STEPS_CAP:-}"
EXTRA_FLAGS=(--max-workers "$MAX_WORKERS")
[ -n "$N_ITER_CAP" ]         && EXTRA_FLAGS+=(--n-iter-cap         "$N_ITER_CAP")
[ -n "$N_RESTARTS_CAP" ]     && EXTRA_FLAGS+=(--n-restarts-cap     "$N_RESTARTS_CAP")
[ -n "$N_TRAIN_CAP" ]        && EXTRA_FLAGS+=(--n-train-cap        "$N_TRAIN_CAP")
[ -n "$N_TEST_CAP" ]         && EXTRA_FLAGS+=(--n-test-cap         "$N_TEST_CAP")
[ -n "$ROLLOUT_STEPS_CAP" ]  && EXTRA_FLAGS+=(--rollout-steps-cap  "$ROLLOUT_STEPS_CAP")
N_CFG=${#CONFIGS[@]}

echo "============================================================"
echo "  Lorenz Statistical Validation Suite"
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
echo "  All ${N_CFG} Lorenz config(s) complete."
echo "============================================================"
