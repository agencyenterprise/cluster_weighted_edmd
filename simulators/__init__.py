"""
Dynamical system simulators for validation experiments.

This sub-package contains the two test systems used to validate
residual-aware Bayesian clustering.  Each module exposes a common
interface:

- ``f(state)`` -- vector field (returns dx/dt).
- ``J(state)`` -- analytic Jacobian of the vector field.
- A data-generation function that returns dicts with keys ``'X'``
  (phase points) and ``'F'`` (vector-field evaluations), ready
  to be wrapped in ``torch.tensor`` and passed to the EM fitters.

Available systems
-----------------
- ``lorenz`` -- Lorenz-63 chaotic attractor (d=3, polynomial RHS).
  Uses ``generate_data()`` to produce a single long trajectory on the
  attractor after discarding a warmup transient.
- ``pendulum`` -- Damped pendulum (d=2, non-polynomial sin(theta)).
  Uses ``sample_phase_space()`` for uniform phase-space regression data
  and ``generate_trajectory()`` for test rollouts.
- ``duffing`` -- Forced double-well Duffing oscillator (Mauroy-Mezic-
  Moehlis lineage testbed).  Provides both an *unforced* 2D variant
  with a saddle-separated two-basin topology and a *forced* 3D
  phase-augmented variant with the canonical chaotic strange attractor.
  Also exposes ``classify_basin()`` for ground-truth basin labels.

Usage
-----
::

    from residual_aware_clustering.simulators import lorenz, pendulum, duffing

    lorenz_data    = lorenz.generate_data(n_steps=5000)
    pendulum_data  = pendulum.sample_phase_space(n_samples=4000)
    duffing_data   = duffing.sample_phase_space(n_samples=4000)
    duffing_forced = duffing.sample_phase_space_forced(n_samples=8000)
"""
