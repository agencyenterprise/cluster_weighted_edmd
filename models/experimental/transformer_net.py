"""
TransformerNetModel -- transformer-based local model for the generic EM pipeline.

Implements the ``LocalModel`` protocol using a small transformer with
self-attention blocks. Each cluster gets its own ``LocalTransformer`` network
that learns local X -> Y mappings around the cluster center with weighted MSE
loss, Adam optimization, and early stopping. Best suited for capturing complex
nonlinear dynamics in high-dimensional spaces where the self-attention
mechanism can model feature interactions more expressively than a plain MLP.

The transformer operates on single state vectors (no sequence dimension) --
each (d,)-dimensional input is projected to ``d_model`` dimensions, passed
through ``n_layers`` pre-norm transformer blocks (multi-head self-attention +
FFN with SiLU activation), and projected back to d dimensions.

Uses the same ``predict_offset`` convention as ``NeuralNetModel``: for
discrete models (default), the network learns ``Y - center``; for continuous
models, it learns the velocity directly.

Usage
-----
Standalone::

    import torch
    from residual_aware_clustering.models.experimental import TransformerNetModel

    model = TransformerNetModel(
        d_model=64,           # internal transformer dimension
        n_heads=4,            # number of attention heads
        d_ff=128,             # feed-forward hidden dimension
        n_layers=2,           # number of transformer blocks
        lr=1e-3,              # Adam learning rate
        n_epochs=80,          # max training epochs per EM iteration
        patience=10,          # early stopping patience
        predict_offset=True,  # discrete mode
    )

    X = torch.randn(500, 32)        # (P, d) current states
    Y = torch.randn(500, 32)        # (P, d) next states
    weights = torch.ones(500)
    center = X.mean(dim=0)

    model.fit(X, Y, weights, center)
    Y_pred = model.predict(X, center)  # (500, 32) predicted next states

With the generic EM pipeline::

    from residual_aware_clustering.models.experimental import (
        generic_em, TransformerNetModel,
    )

    prototype = TransformerNetModel(d_model=64, n_heads=4, n_layers=2)
    state, r, history = generic_em.fit(
        X, X_next, N=5, hp=hp, model_prototype=prototype,
    )

Key concepts
------------
- **Architecture**: ``LocalTransformer`` consists of a linear input projection,
  ``n_layers`` ``LocalTransformerBlock`` modules (pre-norm self-attention + FFN),
  a final LayerNorm, and a linear output projection. Despite operating on
  single vectors (no sequence), the multi-head attention computes feature
  interactions across the ``d_model`` embedding dimensions.
- **Pre-norm design**: Each block applies LayerNorm before attention and before
  the FFN, following the modern transformer convention for more stable training.
- **Weighted loss**: Same weighted MSE as ``NeuralNetModel`` -- each sample is
  scaled by its soft-assignment weight from the EM E-step.
- **Early stopping**: Training halts when loss has not improved for ``patience``
  consecutive epochs; the best network state is restored.
- **Float32 training**: Trains in float32 for GPU efficiency, then converts to
  the EM dtype (float64) for prediction consistency.
- **When to use**: Prefer this over ``NeuralNetModel`` when the dynamics have
  complex feature interactions that benefit from self-attention, or when you
  have enough data per cluster to justify the larger parameter count. For
  simpler dynamics or smaller clusters, ``NeuralNetModel`` or
  ``PolynomialDiscreteEDMD`` may be more data-efficient.
"""

from __future__ import annotations
from typing import Any
import math

import torch
import torch.nn as nn


class LocalTransformerBlock(nn.Module):
    """Single transformer block: self-attention + FFN with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """Initialize a single pre-norm transformer block.

        Parameters
        ----------
        d_model : int
            Model embedding dimension.
        n_heads : int
            Number of attention heads.
        d_ff : int
            Feed-forward hidden dimension.
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and FFN with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape ``(B, d_model)``.

        Returns
        -------
        torch.Tensor
            Output embeddings, shape ``(B, d_model)``.
        """
        # x: (B, d_model) — single token per sample (no sequence dim)
        # Add a sequence dim of 1 for attention
        B, D = x.shape
        H, d_h = self.n_heads, self.d_head

        # Pre-norm attention
        h = self.norm1(x)
        q = self.W_Q(h).view(B, 1, H, d_h).transpose(1, 2)
        k = self.W_K(h).view(B, 1, H, d_h).transpose(1, 2)
        v = self.W_V(h).view(B, 1, H, d_h).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_h)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, D)
        x = x + self.W_O(out)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class LocalTransformer(nn.Module):
    """Small transformer operating on single vectors (no sequence dimension)."""

    def __init__(self, d_input: int, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, d_output: int):
        """Initialize the local transformer network.

        Parameters
        ----------
        d_input : int
            Input dimension.
        d_model : int
            Internal transformer embedding dimension.
        n_heads : int
            Number of attention heads per block.
        d_ff : int
            Feed-forward hidden dimension per block.
        n_layers : int
            Number of transformer blocks.
        d_output : int
            Output dimension.
        """
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([
            LocalTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through input projection, transformer blocks, and output projection.

        Parameters
        ----------
        x : torch.Tensor
            Input vectors, shape ``(B, d_input)``.

        Returns
        -------
        torch.Tensor
            Output vectors, shape ``(B, d_output)``.
        """
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(self.norm_out(h))


class TransformerNetModel:
    """Transformer local model for use with the generic EM pipeline."""

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        d_ff: int = 64,
        n_layers: int = 1,
        lr: float = 1e-3,
        n_epochs: int = 50,
        patience: int = 10,
        predict_offset: bool = True,
        min_samples: int = 10,
    ):
        """Initialize a transformer-based local model.

        Parameters
        ----------
        d_model : int
            Internal transformer embedding dimension.
        n_heads : int
            Number of attention heads per block.
        d_ff : int
            Feed-forward hidden dimension per block.
        n_layers : int
            Number of transformer blocks.
        lr : float
            Adam learning rate.
        n_epochs : int
            Maximum training epochs per fit call.
        patience : int
            Early stopping patience (epochs without improvement).
        predict_offset : bool
            If True (discrete mode), learn ``Y - center``; if False
            (continuous mode), learn the velocity directly.
        min_samples : int
            Minimum effective sample size to attempt training.
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.predict_offset = predict_offset
        self.min_samples = min_samples
        self._net: LocalTransformer | None = None
        self._d: int | None = None

    @property
    def min_points(self) -> int:
        """Minimum effective sample size required to fit.

        Returns
        -------
        int
            The configured ``min_samples`` threshold.
        """
        return self.min_samples

    def _build_net(self, d: int) -> LocalTransformer:
        """Construct the ``LocalTransformer`` architecture.

        Parameters
        ----------
        d : int
            Input and output dimension (state-space dimension).

        Returns
        -------
        LocalTransformer
            Newly created transformer network.
        """
        return LocalTransformer(
            d_input=d, d_model=self.d_model, n_heads=self.n_heads,
            d_ff=self.d_ff, n_layers=self.n_layers, d_output=d,
        )

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            weights: torch.Tensor, center: torch.Tensor) -> None:
        """Train the transformer on weighted data with early stopping.

        Parameters
        ----------
        X : torch.Tensor
            Input states, shape ``(P, d)``.
        Y : torch.Tensor
            Targets (next states or velocities), shape ``(P, d)``.
        weights : torch.Tensor
            Per-sample soft-assignment weights, shape ``(P,)``.
        center : torch.Tensor
            Cluster center, shape ``(d,)``.
        """
        d = X.shape[1]
        self._d = d
        dev, dt = X.device, X.dtype

        U = (X - center).float()
        if self.predict_offset:
            target = (Y - center).float()
        else:
            target = Y.float()
        w = weights.float()

        if w.sum().item() < self.min_samples:
            self._net = self._build_net(d).to(U.device)
            return

        self._net = self._build_net(d).to(U.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        self._net.train()
        for _ in range(self.n_epochs):
            pred = self._net(U)
            residual = (pred - target) ** 2
            loss = (w.unsqueeze(1) * residual).sum() / (w.sum() * d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._net = self._net.to(dtype=dt)

    def predict(self, X: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """Predict outputs using the trained transformer.

        Parameters
        ----------
        X : torch.Tensor
            Input states, shape ``(P, d)``.
        center : torch.Tensor
            Cluster center, shape ``(d,)``.

        Returns
        -------
        torch.Tensor
            Predictions, shape ``(P, d)``. Includes center offset if discrete mode.
        """
        self._net.eval()
        with torch.no_grad():
            U = X - center
            out = self._net(U)
        if self.predict_offset:
            return center + out
        return out

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary of model parameters.

        Returns
        -------
        dict[str, Any]
            Contains 'net_state', 'd', and hyperparameters.
        """
        return {
            'net_state': self._net.state_dict() if self._net else None,
            'd': self._d,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'n_layers': self.n_layers,
            'lr': self.lr,
            'n_epochs': self.n_epochs,
            'predict_offset': self.predict_offset,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore model from a state dictionary.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary previously returned by ``state_dict()``.
        """
        self._d = state['d']
        if state['net_state'] is not None:
            self._net = self._build_net(self._d)
            self._net.load_state_dict(state['net_state'])

    def clone(self) -> TransformerNetModel:
        """Return a fresh instance with the same hyperparameters but no fitted state.

        Returns
        -------
        TransformerNetModel
            New unfitted model with identical configuration.
        """
        return TransformerNetModel(
            d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff,
            n_layers=self.n_layers, lr=self.lr, n_epochs=self.n_epochs,
            patience=self.patience,
            predict_offset=self.predict_offset, min_samples=self.min_samples,
        )

    def fallback_init(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize the network without training as a safe fallback.

        Parameters
        ----------
        d : int
            State-space dimension.
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.
        """
        self._d = d
        self._net = self._build_net(d).to(device=device, dtype=dtype)

    def to(self, device: torch.device, dtype: torch.dtype) -> TransformerNetModel:
        """Move model parameters to the given device and dtype.

        Parameters
        ----------
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.

        Returns
        -------
        TransformerNetModel
            Self, for method chaining.
        """
        if self._net is not None:
            self._net = self._net.to(device=device, dtype=dtype)
        return self
