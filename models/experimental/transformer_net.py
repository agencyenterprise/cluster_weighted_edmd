"""
Transformer local model wrapper for the generic EM pipeline.

Trains a small transformer per cluster to learn X -> Y mappings locally
around each cluster center. Uses the same predict_offset convention as
NeuralNetModel: for discrete models the network learns the offset.
"""

from __future__ import annotations
from typing import Any
import math

import torch
import torch.nn as nn


class LocalTransformerBlock(nn.Module):
    """Single transformer block: self-attention + FFN with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
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
        super().__init__()
        self.proj_in = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([
            LocalTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.min_samples

    def _build_net(self, d: int) -> LocalTransformer:
        return LocalTransformer(
            d_input=d, d_model=self.d_model, n_heads=self.n_heads,
            d_ff=self.d_ff, n_layers=self.n_layers, d_output=d,
        )

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            weights: torch.Tensor, center: torch.Tensor) -> None:
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
        self._net.eval()
        with torch.no_grad():
            U = X - center
            out = self._net(U)
        if self.predict_offset:
            return center + out
        return out

    def state_dict(self) -> dict[str, Any]:
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
        self._d = state['d']
        if state['net_state'] is not None:
            self._net = self._build_net(self._d)
            self._net.load_state_dict(state['net_state'])

    def clone(self) -> TransformerNetModel:
        return TransformerNetModel(
            d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff,
            n_layers=self.n_layers, lr=self.lr, n_epochs=self.n_epochs,
            patience=self.patience,
            predict_offset=self.predict_offset, min_samples=self.min_samples,
        )

    def fallback_init(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        self._d = d
        self._net = self._build_net(d).to(device=device, dtype=dtype)

    def to(self, device: torch.device, dtype: torch.dtype) -> TransformerNetModel:
        if self._net is not None:
            self._net = self._net.to(device=device, dtype=dtype)
        return self
