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
    python -m validation.run_statistical --config "$cfg"
    echo ""
done

echo "============================================================"
echo "  All ${N_CFG} Lorenz config(s) complete."
echo "============================================================"
