"""
Centralized output paths for figures and data.
All paths are relative to the project root (residual_aware_clustering/).
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_ROOT, "papers", "figures")
DATA_DIR = os.path.join(_ROOT, "papers", "data")


def fig_path(filename: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, filename)


def data_path(filename: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, filename)
