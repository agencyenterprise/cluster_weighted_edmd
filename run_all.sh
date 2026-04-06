#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "============================================================"
echo "  Residual-Aware Bayesian Clustering — Full Experiment Suite"
echo "============================================================"
echo ""

# ── Lorenz experiments ────────────────────────────────────────────
echo "[1/6] Lorenz: full experiment (sanity + fit + model selection)"
python -m validation.validation_lorenz
echo ""

echo "[2/6] Lorenz: one-step + rollout vs global EDMD"
python -m validation.validation_lorenz_vs_edmd
echo ""

echo "[3/6] Lorenz: sweep N (all methods)"
python -m validation.validation_lorenz_sweep_N
echo ""

echo "[4/6] Lorenz: Taylor-analytic vs Taylor-LS vs GMM"
python -m validation.validation_lorenz_hybrid
echo ""

echo "[5/6] Lorenz: local EDMD vs global EDMD"
python -m validation.validation_lorenz_local_edmd
echo ""

# ── Pendulum experiments ──────────────────────────────────────────
echo "[6/6] Pendulum: all four method families"
python -m validation.validation_pendulum
echo ""

echo "============================================================"
echo "  All experiments complete."
echo "  Figures: papers/figures/"
echo "  Data:    papers/data/"
echo "============================================================"
