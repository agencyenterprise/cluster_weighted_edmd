#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "============================================================"
echo "  Setting up residual_aware_clustering"
echo "============================================================"

echo ""
echo "[1/3] Installing package in editable mode..."
pip install -e .

echo ""
echo "[2/3] Installing pykoopman extras..."
pip install -e ".[pykoopman]"

echo ""
echo "[3/3] Patching pykoopman for sklearn compatibility..."
python patches/fix_pykoopman_sklearn.py

echo ""
echo "============================================================"
echo "  Setup complete."
echo ""
echo "  Library usage:"
echo "    from residual_aware_clustering import fit_taylor, make_hp"
echo ""
echo "  Run experiments:"
echo "    python -m validation.run_statistical"
echo "============================================================"
