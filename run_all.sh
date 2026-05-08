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
echo "  Systems: ${SYSTEMS[*]}"
echo "  Total configs: ${N_CFG}"
echo "============================================================"

idx=0
for cfg in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    echo ""
    echo "############################################################"
    echo "##  [${idx}/${N_CFG}] ${cfg}"
    echo "############################################################"
    python -m validation.run_statistical --config "$cfg"
    echo ""
done

echo "============================================================"
echo "  All ${N_CFG} configs complete."
echo "  Figures: papers/figures/<config_name>.png"
echo "  Data:    papers/data/<config_name>.json"
echo "============================================================"
