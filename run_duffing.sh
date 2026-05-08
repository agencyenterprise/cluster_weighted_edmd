#!/usr/bin/env bash
#
# Multi-seed, multi-config statistical validation of the residual-aware
# clustering pipeline on the unforced double-well Duffing oscillator.
#
# Iterates over every YAML config in ``config/duffing/`` (or a custom set
# supplied as positional args) and writes one JSON + one PNG per config
# to ``papers/data/`` and ``papers/figures/`` respectively.  Each config
# uses the 10-seed default unless overridden inside the YAML.
#
# Usage
# -----
#   ./run_duffing.sh                                  # all configs
#   ./run_duffing.sh config/duffing/uniform_baseline.yaml
#   ./run_duffing.sh config/duffing/uniform_*.yaml    # glob
#
# Outputs
# -------
#   papers/data/<config_name>.json          per-seed metrics
#   papers/data/<config_name>_models.pt     fitted models for best seed
#   papers/figures/<config_name>.png        error-bar comparison plot
#
# Each config's outputs are namespaced by its `name:` field, so multiple
# configs do NOT overwrite each other.

set -e
cd "$(dirname "$0")"

# -- Choose configs -----------------------------------------------------------

if [ $# -eq 0 ]; then
    CONFIGS=(config/duffing/*.yaml)
else
    CONFIGS=("$@")
fi

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

# Speed knobs (override via env vars).
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
echo "  Duffing Statistical Validation Suite"
echo "  Run ID:        ${RUN_ID}"
echo "  Output dir:    papers/data/${RUN_ID}/   papers/figures/${RUN_ID}/"
echo "  ${N_CFG} config(s) to run:"
for cfg in "${CONFIGS[@]}"; do
    echo "    - ${cfg}"
done
echo "============================================================"
echo ""

# -- Run each config ----------------------------------------------------------

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

echo ""
echo "============================================================"
echo "  All ${N_CFG} Duffing config(s) complete (Run ID: ${RUN_ID})."
echo "  Data:    papers/data/${RUN_ID}/<config_name>.json"
echo "  Figures: papers/figures/${RUN_ID}/<config_name>.png"
echo "============================================================"
