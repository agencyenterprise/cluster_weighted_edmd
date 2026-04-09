"""
Observable transformations for EDMD lifting.

Provides time-delay embeddings and block Hankel matrix construction
compatible with the CWM pipeline. The caller builds delay-embedded
(X, Y) pairs, then passes them to the EM pipeline as usual.

Convention matches pykoopman: rows are samples (time steps),
columns are features (delay-embedded state variables).
"""

from __future__ import annotations
from enum import Enum

import torch
import numpy as np


class TimeDelay:
    """Time-delay observable transformation.

    For a trajectory [h_0, h_1, ..., h_T] with n_delays=2 and delay=1,
    the transformed output at time t is:
        [h(t), h(t-1), h(t-2)]

    For multiple state variables [x₁, x₂] with n_delays=2:
        [x₁(t), x₂(t), x₁(t-1), x₂(t-1), x₁(t-2), x₂(t-2)]

    This is the block Hankel structure in row-based format.

    Args:
        delay: Spacing between time-lagged copies (in samples). Default: 1.
        n_delays: Number of past states to include. Default: 2.
    """

    def __init__(self, delay: int = 1, n_delays: int = 2):
        if delay < 1:
            raise ValueError("delay must be a positive int")
        if n_delays < 1:
            raise ValueError("n_delays must be a positive int")

        self.delay = delay
        self.n_delays = n_delays

    @property
    def n_consumed_samples(self) -> int:
        """Number of samples lost due to insufficient history."""
        return self.delay * self.n_delays

    def n_output_features(self, n_input_features: int) -> int:
        """Output dimension given input dimension."""
        return n_input_features * (1 + self.n_delays)

    def transform(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Apply time-delay embedding to a single trajectory.

        Args:
            x: (T, d) sequential data. Rows must be ordered in time.

        Returns:
            (T - delay*n_delays, d*(1+n_delays)) delay-embedded data.
        """
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            device, dtype = x.device, x.dtype
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        T, d = x_np.shape
        consumed = self.n_consumed_samples
        if T < consumed + 1:
            raise ValueError(
                f"Trajectory too short ({T} samples) for delay={self.delay}, "
                f"n_delays={self.n_delays}. Need at least {consumed + 1}.")

        n_out = T - consumed
        d_out = d * (1 + self.n_delays)
        y = np.empty((n_out, d_out), dtype=x_np.dtype)

        # Current state
        y[:, :d] = x_np[consumed:]

        # Delayed states
        for k in range(1, self.n_delays + 1):
            offset = consumed - k * self.delay
            y[:, k * d:(k + 1) * d] = x_np[offset:offset + n_out]

        if is_torch:
            return torch.tensor(y, device=device, dtype=dtype)
        return y

    def transform_trajectories(
        self, trajectories: list[torch.Tensor | np.ndarray]
    ) -> torch.Tensor | np.ndarray:
        """Apply time-delay embedding to multiple trajectories, respecting boundaries.

        Args:
            trajectories: List of (T_i, d) sequential data arrays.

        Returns:
            Concatenated delay-embedded data from all trajectories.
        """
        results = []
        for traj in trajectories:
            if (len(traj) if isinstance(traj, np.ndarray) else traj.shape[0]) > self.n_consumed_samples:
                results.append(self.transform(traj))

        if not results:
            raise ValueError("No trajectories long enough for the given delay parameters.")

        if isinstance(results[0], torch.Tensor):
            return torch.cat(results, dim=0)
        return np.concatenate(results, axis=0)

    def build_pairs(
        self, x: torch.Tensor | np.ndarray
    ) -> tuple:
        """Build (X, Y) consecutive pairs from a delay-embedded trajectory.

        Given a trajectory, constructs delay-embedded X at time t and Y at time t+1.

        Args:
            x: (T, d) sequential data.

        Returns:
            (X, Y) where X is (N, d_out) at time t and Y is (N, d_out) at time t+1.
            d_out = d * (1 + n_delays).
        """
        embedded = self.transform(x)
        X = embedded[:-1]
        Y = embedded[1:]
        return X, Y

    def build_pairs_from_trajectories(
        self, trajectories: list[torch.Tensor | np.ndarray]
    ) -> tuple:
        """Build (X, Y) pairs from multiple trajectories, respecting boundaries.

        Args:
            trajectories: List of (T_i, d) sequential data arrays.

        Returns:
            (X, Y) concatenated pairs from all trajectories.
        """
        all_X, all_Y = [], []
        for traj in trajectories:
            T = len(traj) if isinstance(traj, np.ndarray) else traj.shape[0]
            if T > self.n_consumed_samples + 1:
                X, Y = self.build_pairs(traj)
                all_X.append(X)
                all_Y.append(Y)

        if not all_X:
            raise ValueError("No trajectories long enough for the given delay parameters.")

        if isinstance(all_X[0], torch.Tensor):
            return torch.cat(all_X, dim=0), torch.cat(all_Y, dim=0)
        return np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0)
