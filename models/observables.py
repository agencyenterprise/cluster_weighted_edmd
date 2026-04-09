"""
Observable transformations for EDMD lifting.

Time-delay embeddings (block Hankel) for state-space augmentation.
Given a low-dimensional trajectory, delay embedding reconstructs a
higher-dimensional representation that captures temporal structure —
critical when the observed state is a partial observation of the
underlying dynamical system (Takens' theorem).

Usage
-----
Single trajectory::

    from residual_aware_clustering.models.observables import TimeDelay

    # Raw trajectory: T timesteps, d state variables
    # e.g. hidden states from a transformer, shape (200, 576)
    trajectory = ...  # (T, d) numpy array or torch tensor

    # Embed with 3 delay copies, spaced 1 timestep apart
    td = TimeDelay(delay=1, n_delays=3)

    # Transform: augments each timestep with its past
    # Input:  (T, d)
    # Output: (T - 3, d * 4)  — current + 3 past copies
    embedded = td.transform(trajectory)

    # Build consecutive pairs for EDMD fitting
    # X[i] = embedded state at time t
    # Y[i] = embedded state at time t+1
    X, Y = td.build_pairs(trajectory)

Multiple trajectories (respects episode boundaries)::

    trajectories = [traj1, traj2, traj3]  # list of (T_i, d) arrays
    X, Y = td.build_pairs_from_trajectories(trajectories)

Parameters explained::

    TimeDelay(delay=1, n_delays=3)

    delay    = spacing between snapshots (in timesteps)
    n_delays = number of past snapshots to include

    For a 2D state [x₁, x₂] with delay=1, n_delays=2:

    Input trajectory:
        t=0: [x₁(0), x₂(0)]
        t=1: [x₁(1), x₂(1)]
        t=2: [x₁(2), x₂(2)]
        t=3: [x₁(3), x₂(3)]
        t=4: [x₁(4), x₂(4)]

    Output (after consuming 2 samples for history):
        t=2: [x₁(2), x₂(2), x₁(1), x₂(1), x₁(0), x₂(0)]
        t=3: [x₁(3), x₂(3), x₁(2), x₂(2), x₁(1), x₂(1)]
        t=4: [x₁(4), x₂(4), x₁(3), x₂(3), x₁(2), x₂(2)]
              ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
              current state   1 step back     2 steps back

    With delay=3, n_delays=2, the spacing is 3 timesteps:
        t=6: [x₁(6), x₂(6), x₁(3), x₂(3), x₁(0), x₂(0)]
              current state   3 steps back    6 steps back

Integration with CWM pipeline::

    from residual_aware_clustering.models.observables import TimeDelay
    from residual_aware_clustering.models.em_local_edmd_discrete import fit as fit_local_edmd_disc

    td = TimeDelay(delay=1, n_delays=3)
    X, Y = td.build_pairs(trajectory)

    # X and Y are now delay-embedded — pass directly to EDMD
    state, elbos, labels = fit_local_edmd_disc(
        X, Y, N=10, hp=hyperparams, degree=1, n_iter=100
    )

Convention: matches pykoopman's TimeDelay ordering (most recent first).
"""

from __future__ import annotations

import torch
import numpy as np


class TimeDelay:
    """Time-delay observable: augments each state with its recent past.

    Constructs a block Hankel embedding from sequential data.
    Output at time t is [h(t), h(t-delay), h(t-2*delay), ...].

    Args:
        delay: Spacing between time-lagged copies, in timesteps.
            delay=1 means consecutive steps; delay=3 means every 3rd step.
        n_delays: Number of past snapshots to include (excluding current).
            n_delays=2 with delay=1 gives [h(t), h(t-1), h(t-2)].

    Properties:
        n_consumed_samples: Number of initial timesteps lost = delay * n_delays.
        n_output_features(d): Output dimension = d * (1 + n_delays).
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
        """Number of initial timesteps lost due to insufficient history."""
        return self.delay * self.n_delays

    def n_output_features(self, n_input_features: int) -> int:
        """Output dimension: n_input_features * (1 + n_delays)."""
        return n_input_features * (1 + self.n_delays)

    def transform(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Apply time-delay embedding to a single trajectory.

        Args:
            x: (T, d) sequential data, rows ordered in time.

        Returns:
            (T - n_consumed_samples, d * (1 + n_delays)) embedded data.
            Accepts and returns the same type (numpy or torch).
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
        """Apply time-delay embedding to multiple trajectories.

        Respects trajectory boundaries — no cross-contamination between
        episodes. Trajectories shorter than n_consumed_samples are skipped.

        Args:
            trajectories: List of (T_i, d) arrays, each an independent episode.

        Returns:
            Concatenated delay-embedded data from all valid trajectories.
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
        """Build consecutive (X, Y) pairs from a delay-embedded trajectory.

        X[i] is the embedded state at time t, Y[i] at time t+1.
        These pairs are ready to pass directly to discrete EDMD fitting.

        Args:
            x: (T, d) sequential data.

        Returns:
            (X, Y) each of shape (T - n_consumed_samples - 1, d * (1 + n_delays)).
        """
        embedded = self.transform(x)
        X = embedded[:-1]
        Y = embedded[1:]
        return X, Y

    def build_pairs_from_trajectories(
        self, trajectories: list[torch.Tensor | np.ndarray]
    ) -> tuple:
        """Build (X, Y) pairs from multiple trajectories.

        Respects trajectory boundaries — pairs never cross episodes.
        Trajectories too short to produce at least one pair are skipped.

        Args:
            trajectories: List of (T_i, d) arrays, each an independent episode.

        Returns:
            (X, Y) concatenated pairs from all valid trajectories.
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
