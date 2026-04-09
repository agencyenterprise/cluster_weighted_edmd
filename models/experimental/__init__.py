"""
Experimental: generic EM pipeline with pluggable local surrogate models.

This package provides a single, model-agnostic EM implementation for
residual-aware clustering. Choose a local model type, create a prototype
instance, and pass it to ``generic_em.fit()`` -- the pipeline handles
cloning, initialization, fitting, and pruning automatically.

Available classes
-----------------
``LocalModel``
    Protocol (interface) that all local surrogate models must implement.
    See ``local_model.py`` for the full method contract.

``PolynomialDiscreteEDMD``
    Discrete Koopman operator via polynomial-lifted EDMD. Supports FULL
    and DIAGONAL observable types. Best for low-to-moderate dimensions.

``PolynomialContinuousEDMD``
    Continuous Koopman generator for velocity field fitting. Same polynomial
    lifting but fits f(x) instead of x_{t+1}.

``NeuralNetModel``
    MLP-based local model with early stopping. Good for high-dimensional
    data where polynomial lifting is too expensive.

``TransformerNetModel``
    Transformer-based local model with self-attention blocks. Best for
    capturing complex nonlinear dynamics in high-dimensional spaces.

``ObservableType``
    Enum for polynomial observable types (FULL, DIAGONAL).

``generic_em``
    Module containing the EM pipeline functions (``fit``, ``initialize``,
    ``e_step``, ``m_step``, ``compute_elbo``, ``prune_dead``).

Usage
-----
::

    from residual_aware_clustering.models.experimental import (
        generic_em, PolynomialDiscreteEDMD, NeuralNetModel, TransformerNetModel,
    )

    # Polynomial EDMD
    state, r, history = generic_em.fit(
        X, X_next, N=8, hp=hp,
        model_prototype=PolynomialDiscreteEDMD(degree=3),
    )

    # Neural network
    state, r, history = generic_em.fit(
        X, X_next, N=5, hp=hp,
        model_prototype=NeuralNetModel(hidden_dims=(128, 128)),
    )

    # Transformer
    state, r, history = generic_em.fit(
        X, X_next, N=5, hp=hp,
        model_prototype=TransformerNetModel(d_model=64, n_heads=4),
    )
"""

from .local_model import LocalModel
from ..em_local_edmd import ObservableType
from .polynomial_discrete import PolynomialDiscreteEDMD
from .polynomial_continuous import PolynomialContinuousEDMD
from .neural_net import NeuralNetModel
from .transformer_net import TransformerNetModel
from . import generic_em

__all__ = [
    "LocalModel",
    "ObservableType",
    "PolynomialDiscreteEDMD",
    "PolynomialContinuousEDMD",
    "NeuralNetModel",
    "TransformerNetModel",
    "generic_em",
]
