"""
Dynamical system simulators.

Each module provides:
- ``f(state)``: vector field (returns dx/dt)
- ``J(state)``: analytic Jacobian
- ``generate_data()`` or ``sample_phase_space()``: data generation

Available systems:
- ``lorenz``: Lorenz attractor (d=3, polynomial RHS)
- ``pendulum``: Damped pendulum (d=2, non-polynomial sin(theta))
"""
