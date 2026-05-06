"""
toy_problem.py
==============
Defines the toy constrained optimisation problem used throughout the tutorial:

    min  f(x, y) = x^2 + y^2
    s.t. h(x, y) = x + y - 1 = 0

Analytical solution: x* = y* = 0.5, lambda* = -1.

All functions are written with JAX so that they are:
  * Fully differentiable — `jax.grad` can differentiate through them.
  * JIT-compilable      — wrapping a call in `jax.jit` compiles it to XLA,
                           giving near-native performance on CPU/GPU/TPU.
  * Vmappable           — `jax.vmap` can batch calls over arrays of inputs
                           without explicit Python loops.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Problem functions
# ---------------------------------------------------------------------------

def objective(xy: jnp.ndarray) -> jnp.ndarray:
    """Computes the objective function f(x, y) = x² + y².

    This is a convex quadratic — its unconstrained minimum is at the origin
    (0, 0), but the equality constraint x + y = 1 shifts the optimum to
    (0.5, 0.5).

    Args:
        xy: Primal variable vector of shape ``(2,)`` containing [x, y].

    Returns:
        A scalar JAX array with the value of f(x, y).
    """
    x, y = xy[0], xy[1]
    return x ** 2 + y ** 2


def equality_constraint(xy: jnp.ndarray) -> jnp.ndarray:
    """Evaluates the equality constraint h(x, y) = x + y − 1.

    A feasible point must satisfy h(x, y) = 0, i.e. x + y = 1.  The
    Lagrangian penalises any deviation from this affine hyperplane via the
    multiplier λ.

    Args:
        xy: Primal variable vector of shape ``(2,)`` containing [x, y].

    Returns:
        A scalar JAX array with the constraint residual (0 when satisfied).
    """
    x, y = xy[0], xy[1]
    return x + y - 1.0


def lagrangian(xy: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """Evaluates the Lagrangian L(x, y, λ) = f(x, y) + λ · h(x, y).

    The Lagrangian combines the objective with a weighted penalty for the
    equality constraint.  At the optimum, stationarity (∇_x L = 0) together
    with primal feasibility (h = 0) characterise the KKT conditions.

    Note: This problem has no inequality constraints, so there is no μ term.

    Args:
        xy: Primal variable vector of shape ``(2,)`` containing [x, y].
        lam: Lagrange multiplier array of shape ``(1,)`` for the equality
            constraint.

    Returns:
        A scalar JAX array with the Lagrangian value.
    """
    # Weighted sum: objective + lambda * constraint residual
    return objective(xy) + lam[0] * equality_constraint(xy)


# ---------------------------------------------------------------------------
# Gradient-flow right-hand sides  (KKT → IVP)
# ---------------------------------------------------------------------------

def primal_rhs(xy: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """Computes the primal gradient-flow RHS: dx/dt = −∇_x L.

    Driving the primal variables along the *negative* gradient of the
    Lagrangian w.r.t. x is equivalent to minimising L w.r.t. x.  At steady
    state (dx/dt = 0), the stationarity KKT condition ∇_x L = 0 holds.

    ``jax.grad(lagrangian, argnums=0)`` differentiates ``lagrangian`` with
    respect to its *first* argument (xy) while treating all other arguments
    as constants.  This gives the exact gradient ∇_x L without finite
    differences.

    Args:
        xy: Primal variable vector of shape ``(2,)``.
        lam: Dual variable array of shape ``(1,)``.

    Returns:
        Array of shape ``(2,)`` representing dx/dt.
    """
    # jax.grad returns a function; calling it with (xy, lam) gives ∇_x L
    grad_xy = jax.grad(lagrangian, argnums=0)(xy, lam)
    return -grad_xy   # gradient *descent* on the primal variables


def dual_rhs(xy: jnp.ndarray) -> jnp.ndarray:
    """Computes the dual gradient-flow RHS: dλ/dt = ∇_λ L = h(x).

    Since L = f(x) + λ h(x), we have ∂L/∂λ = h(x).  Driving λ along this
    gradient pushes it towards values that enforce primal feasibility:
    at steady state (dλ/dt = 0), h(x) = 0 must hold.

    Args:
        xy: Primal variable vector of shape ``(2,)``.

    Returns:
        Array of shape ``(1,)`` representing dλ/dt.
    """
    # Wrap the scalar in a length-1 array to keep shapes consistent with
    # the state vector z = [x, y, lambda]
    return jnp.array([equality_constraint(xy)])


# ---------------------------------------------------------------------------
# Analytical solution (for verification / plotting reference lines)
# ---------------------------------------------------------------------------

ANALYTICAL_SOLUTION = {
    "x": 0.5,
    "y": 0.5,
    "lambda": -1.0,
}

