#!/usr/bin/env bash
#
# Run every YAML config across every system: the full paper-grade
# statistical validation suite.  By default this runs all 36 configs
# (12 per system x 3 systems).
#
# Usage
# -----
#   ./run_all.sh                  # all configs, all systems
#   ./run_all.sh duffing pendulum # only those two systems
#
# Outputs land in papers/data/<config_name>.json and
# papers/figures/<config_name>.png. Each config's `name:` field is used
# as the output stem so nothing overwrites.

set -e
cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
    SYSTEMS=(lorenz pendulum duffing)
else
    SYSTEMS=("$@")
fi

# Run identifier: outputs land under papers/data/$RUN_ID/ and papers/figures/$RUN_ID/.
# Auto-pick a timestamp so back-to-back runs don't overwrite each other.
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

# Speed knobs (override via env vars). Defaults pick a sensible parallelism.
MAX_WORKERS="${MAX_WORKERS:-0}"               # 0 = auto-pick min(cpu_count, n_seeds)
N_ITER_CAP="${N_ITER_CAP:-}"                  # cap EM iterations per restart
N_RESTARTS_CAP="${N_RESTARTS_CAP:-}"          # cap number of EM restarts
N_TRAIN_CAP="${N_TRAIN_CAP:-}"                # cap n_train per system (huge speedup)
N_TEST_CAP="${N_TEST_CAP:-}"                  # cap n_test per system
ROLLOUT_STEPS_CAP="${ROLLOUT_STEPS_CAP:-}"    # cap rollout_steps per system

EXTRA_FLAGS=(--max-workers "$MAX_WORKERS" --run-id "$RUN_ID")
[ -n "$N_ITER_CAP" ]         && EXTRA_FLAGS+=(--n-iter-cap         "$N_ITER_CAP")
[ -n "$N_RESTARTS_CAP" ]     && EXTRA_FLAGS+=(--n-restarts-cap     "$N_RESTARTS_CAP")
[ -n "$N_TRAIN_CAP" ]        && EXTRA_FLAGS+=(--n-train-cap        "$N_TRAIN_CAP")
[ -n "$N_TEST_CAP" ]         && EXTRA_FLAGS+=(--n-test-cap         "$N_TEST_CAP")
[ -n "$ROLLOUT_STEPS_CAP" ]  && EXTRA_FLAGS+=(--rollout-steps-cap  "$ROLLOUT_STEPS_CAP")

CONFIGS=()
for sys in "${SYSTEMS[@]}"; do
    if [ ! -d "config/${sys}" ]; then
        echo "WARNING: no config dir for system '${sys}' (skipping)"
        continue
    fi
    for cfg in config/${sys}/*.yaml; do
        CONFIGS+=("$cfg")
    done
done
N_CFG=${#CONFIGS[@]}

echo "============================================================"
echo "  Residual-Aware Bayesian Clustering -- Full Suite"
echo "  Run ID:         ${RUN_ID}"
echo "  Systems:        ${SYSTEMS[*]}"
echo "  Total configs:  ${N_CFG}"
echo "  Output dir:     papers/data/${RUN_ID}/   papers/figures/${RUN_ID}/"
echo "  Extra flags:    ${EXTRA_FLAGS[*]}"
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
echo "  All ${N_CFG} configs complete (Run ID: ${RUN_ID})."
echo "  Data:    papers/data/${RUN_ID}/<config_name>.json"
echo "  Figures: papers/figures/${RUN_ID}/<config_name>.png"
echo ""
echo "  To compile analysis for this run:"
echo "    python -m validation.analyze_results \\"
echo "      --data-dir papers/data/${RUN_ID} \\"
echo "      --out-dir  papers/analysis/${RUN_ID}"
echo "============================================================"
