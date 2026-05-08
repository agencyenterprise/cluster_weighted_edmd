#!/usr/bin/env bash
#
# End-to-end pipeline smoke test.
#
# Runs the three smoke configs (one per system) through
# validation/run_statistical.py --config, then runs
# validation/analyze_results.py over the freshly produced corpus, and
# verifies that every expected output file exists.
#
# Total runtime: ~1-3 min on a laptop. Use this before kicking off
# ./run_all.sh on runpod.
#
# Usage
# -----
#   ./run_all_smoke.sh                 # run all three smokes + analysis
#   ./run_all_smoke.sh --skip-analyze  # data only (no analysis pass)
#
# Exits non-zero if any expected artifact is missing.

set -e
cd "$(dirname "$0")"

SKIP_ANALYZE=false
for arg in "$@"; do
    case "$arg" in
        --skip-analyze) SKIP_ANALYZE=true ;;
    esac
done

CONFIGS=(
    config/smoke/duffing.yaml
    config/smoke/pendulum.yaml
    config/smoke/lorenz.yaml
)
NAMES=(
    smoke_duffing
    smoke_pendulum
    smoke_lorenz
)

echo "============================================================"
echo "  Pipeline Smoke Test"
echo "  ${#CONFIGS[@]} smoke config(s); analyze step: $([ $SKIP_ANALYZE = true ] && echo skip || echo run)"
echo "============================================================"
echo ""

# -- Step 1: run each smoke config ------------------------------------------

idx=0
for cfg in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    echo "############################################################"
    echo "##  [${idx}/${#CONFIGS[@]}] ${cfg}"
    echo "############################################################"
    python -m validation.run_statistical --config "$cfg"
    echo ""
done

# -- Step 2: verify per-config outputs exist --------------------------------

echo "============================================================"
echo "  Verifying per-config outputs"
echo "============================================================"
fail=0
for name in "${NAMES[@]}"; do
    for path in \
        "papers/data/${name}.json" \
        "papers/data/${name}_models.pt"
    do
        if [ -f "$path" ]; then
            size=$(wc -c < "$path" | tr -d ' ')
            echo "  OK    ${path}  (${size} bytes)"
        else
            echo "  MISSING ${path}"
            fail=$((fail + 1))
        fi
    done
done

if [ $fail -gt 0 ]; then
    echo ""
    echo "FAIL: ${fail} expected per-config artifact(s) missing."
    exit 1
fi

# -- Step 3: run analyzer ---------------------------------------------------

if [ "$SKIP_ANALYZE" = false ]; then
    echo ""
    echo "============================================================"
    echo "  Running analyzer over the smoke corpus"
    echo "============================================================"
    python -m validation.analyze_results

    # -- Step 4: verify analyzer outputs ------------------------------------
    echo ""
    echo "============================================================"
    echo "  Verifying analyzer outputs"
    echo "============================================================"
    for path in \
        "papers/analysis/all_results.csv" \
        "papers/analysis/summary.csv" \
        "papers/analysis/summary_duffing.md" \
        "papers/analysis/summary_duffing.tex" \
        "papers/analysis/summary_pendulum.md" \
        "papers/analysis/summary_pendulum.tex" \
        "papers/analysis/summary_lorenz.md" \
        "papers/analysis/summary_lorenz.tex" \
        "papers/analysis/paired_tests.md"
    do
        if [ -f "$path" ]; then
            size=$(wc -c < "$path" | tr -d ' ')
            echo "  OK    ${path}  (${size} bytes)"
        else
            echo "  MISSING ${path}"
            fail=$((fail + 1))
        fi
    done

    # At least one bars chart and one Pareto plot per system
    for sys in duffing pendulum lorenz; do
        bars=$(ls papers/analysis/bars_${sys}_*.png 2>/dev/null | wc -l | tr -d ' ')
        par=$(ls papers/analysis/pareto_${sys}_*.png 2>/dev/null | wc -l | tr -d ' ')
        if [ "$bars" -ge 1 ] && [ "$par" -ge 1 ]; then
            echo "  OK    papers/analysis/bars_${sys}_*.png   (${bars} files)"
            echo "  OK    papers/analysis/pareto_${sys}_*.png (${par} files)"
        else
            echo "  MISSING bars/pareto PNGs for ${sys} (bars=${bars}, pareto=${par})"
            fail=$((fail + 1))
        fi
    done

    if [ $fail -gt 0 ]; then
        echo ""
        echo "FAIL: ${fail} analyzer artifact(s) missing."
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "  PASS - smoke test complete."
echo "  Per-config data: papers/data/smoke_<system>.{json,_models.pt}"
if [ "$SKIP_ANALYZE" = false ]; then
    echo "  Analysis:        papers/analysis/{summary_*,paired_tests}.{md,tex}"
    echo "                   papers/analysis/{ablation,robustness}_*.png"
fi
echo "============================================================"
