"""
Experimental: generic EM pipeline with pluggable local models.

Usage:
    from residual_aware_clustering.models.experimental import (
        generic_em, PolynomialDiscreteEDMD, PolynomialContinuousEDMD, NeuralNetModel,
    )

    model = PolynomialDiscreteEDMD(degree=3)
    state, r, history = generic_em.fit(X, X_next, N=8, hp=hp, model_prototype=model)
"""

from .local_model import LocalModel
from .polynomial_discrete import PolynomialDiscreteEDMD
from .polynomial_continuous import PolynomialContinuousEDMD
from .neural_net import NeuralNetModel
from .transformer_net import TransformerNetModel
from . import generic_em

__all__ = [
    "LocalModel",
    "PolynomialDiscreteEDMD",
    "PolynomialContinuousEDMD",
    "NeuralNetModel",
    "TransformerNetModel",
    "generic_em",
]
