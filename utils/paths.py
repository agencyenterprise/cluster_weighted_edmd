"""
Centralized output paths for paper figures and experiment data.

All paths are anchored to the project root
(``residual_aware_clustering/``) so that scripts can be run from any
working directory and still write to the correct locations.

Provides two helpers that create the target directory on first call and
return the full path:

- ``fig_path(filename)`` -- path under ``papers/figures/``.
- ``data_path(filename)`` -- path under ``papers/data/``.

Usage
-----
::

    from residual_aware_clustering.utils.paths import fig_path, data_path

    # Save a matplotlib figure
    plt.savefig(fig_path("lorenz_clusters.png"), dpi=150)

    # Save experiment results
    np.save(data_path("elbo_history.npy"), elbo_array)

Key concepts
------------
- **Auto-creation**: both functions call ``os.makedirs(..., exist_ok=True)``
  so the caller never needs to create directories manually.
- **Absolute paths**: built from ``__file__`` so they work regardless
  of the caller's working directory.
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_ROOT, "papers", "figures")
DATA_DIR = os.path.join(_ROOT, "papers", "data")


def fig_path(filename: str) -> str:
    """Return the full path for a figure file under ``papers/figures/``.

    The filename may include intermediate directory components (e.g.
    ``"2026-05-08_15-30-42/lorenz_clusters.png"`` for a per-run subdir);
    every intermediate directory is created on first call.

    Parameters
    ----------
    filename : str
        File name; may contain ``/`` to nest under sub-directories.

    Returns
    -------
    str
        Absolute path to the figure file. The directory tree is created
        as needed.
    """
    full = os.path.join(FIGURES_DIR, filename)
    os.makedirs(os.path.dirname(full) or FIGURES_DIR, exist_ok=True)
    return full


def data_path(filename: str) -> str:
    """Return the full path for a data file under ``papers/data/``.

    The filename may include intermediate directory components (e.g.
    ``"2026-05-08_15-30-42/lorenz_baseline.json"`` for a per-run
    subdir); every intermediate directory is created on first call.

    Parameters
    ----------
    filename : str
        File name; may contain ``/`` to nest under sub-directories.

    Returns
    -------
    str
        Absolute path to the data file. The directory tree is created
        as needed.
    """
    full = os.path.join(DATA_DIR, filename)
    os.makedirs(os.path.dirname(full) or DATA_DIR, exist_ok=True)
    return full
