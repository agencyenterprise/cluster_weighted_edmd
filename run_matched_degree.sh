#!/usr/bin/env bash
#
# Re-run ONLY the configs that have been updated with matched-degree
# Local-EDMD-disc comparisons:
#
#   - config/lorenz/attractor_baseline.yaml   (le_degrees: [2, 3])
#   - config/pendulum/uniform_baseline.yaml   (le_degrees: [2, 4])
#   - config/duffing/uniform_baseline.yaml    (le4_N_list, le5_N_list added)
#
# These are the apples-to-apples partition-vs-global comparison points
# missing from the previous corpus. Outputs land under a fresh, timestamped
# papers/data/<RUN_ID>/ subdirectory so they don't overwrite the existing
# full-corpus run.
#
# Speed knobs hard-coded to the values you specified:
#   N_TRAIN_CAP=2000  ROLLOUT_STEPS_CAP=200  N_RESTARTS_CAP=1
# Override any of them via env var if needed:
#   N_RESTARTS_CAP=2 ./run_matched_degree.sh
#
# Auto-parallelizes seeds; on a 10+ core box this should finish in
# ~20-40 min for all three baselines combined.

set -e
cd "$(dirname "$0")"

# This script writes flat to papers/data/<config_name>.json (no per-run
# subdirectory). NOTE: this WILL overwrite the existing baseline JSONs from
# the previous full-corpus run for the three matched-degree configs:
#   - papers/data/lorenz_attractor_baseline.json
#   - papers/data/pendulum_uniform_baseline.json
#   - papers/data/duffing_uniform_baseline.json
# Other configs from the previous run are untouched.

# Speed knobs (defaults match the user-specified targets; env overrideable)
MAX_WORKERS="${MAX_WORKERS:-0}"               # 0 = auto (min(cpu_count, 10))
N_TRAIN_CAP="${N_TRAIN_CAP:-2000}"
ROLLOUT_STEPS_CAP="${ROLLOUT_STEPS_CAP:-200}"
N_RESTARTS_CAP="${N_RESTARTS_CAP:-1}"
N_ITER_CAP="${N_ITER_CAP:-}"                  # leave empty (use YAML's n_iter)
N_TEST_CAP="${N_TEST_CAP:-}"

# No --run-id: outputs land at papers/data/<config_name>.json (flat).
EXTRA_FLAGS=(--max-workers "$MAX_WORKERS")
EXTRA_FLAGS+=(--n-train-cap        "$N_TRAIN_CAP")
EXTRA_FLAGS+=(--rollout-steps-cap  "$ROLLOUT_STEPS_CAP")
EXTRA_FLAGS+=(--n-restarts-cap     "$N_RESTARTS_CAP")
[ -n "$N_ITER_CAP" ]  && EXTRA_FLAGS+=(--n-iter-cap "$N_ITER_CAP")
[ -n "$N_TEST_CAP" ]  && EXTRA_FLAGS+=(--n-test-cap "$N_TEST_CAP")

# Configs that have the matched-degree settings applied:
CONFIGS=(
    config/lorenz/attractor_baseline.yaml
    config/pendulum/uniform_baseline.yaml
    config/duffing/uniform_baseline.yaml
)

echo "============================================================"
echo "  Matched-degree Local-EDMD-disc re-run"
echo "  Output dir:    papers/data/   papers/figures/   (FLAT, no subdir)"
echo "  Configs:       ${#CONFIGS[@]}"
for cfg in "${CONFIGS[@]}"; do
    echo "    - ${cfg}"
done
echo "  Caps:          n_train=${N_TRAIN_CAP}  rollout_steps=${ROLLOUT_STEPS_CAP}  n_restarts=${N_RESTARTS_CAP}"
echo "                 max_workers=${MAX_WORKERS} (0 = auto)"
echo "  WARNING: this overwrites papers/data/{lorenz,pendulum,duffing}_*_baseline.json"
echo "============================================================"
echo ""

idx=0
for cfg in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    echo ""
    echo "############################################################"
    echo "##  [${idx}/${#CONFIGS[@]}] ${cfg}"
    echo "############################################################"
    python -m validation.run_statistical --config "$cfg" "${EXTRA_FLAGS[@]}"
    echo ""
done

# -- Run analyzer over the (now-updated) papers/data/ corpus ---------------

ANALYSIS_DIR="analysis"
echo ""
echo "============================================================"
echo "  Compiling analysis over papers/data/"
echo "============================================================"
python -m validation.analyze_results \
    --data-dir "papers/data" \
    --out-dir  "${ANALYSIS_DIR}"

echo ""
echo "============================================================"
echo "  PASS - matched-degree re-run complete."
echo "  Per-config data:  papers/data/<config>.json"
echo "  Analysis tables:  ${ANALYSIS_DIR}/summary_<system>.{md,tex}"
echo "  Pareto figures:   ${ANALYSIS_DIR}/pareto_<system>_<metric>.png"
echo "  Bar charts:       ${ANALYSIS_DIR}/bars_<system>_<config>_<metric>.png"
echo ""
echo "  New methods to look for:"
echo "    Lorenz:    Local-EDMD-disc d3 N={5,12,20,50}"
echo "    Pendulum:  Local-EDMD-disc d4 N={2,4,8,16}"
echo "    Duffing:   Local-EDMD-disc d4 N={2,4}, Local-EDMD-disc d5 N={2,4}"
echo "============================================================"
