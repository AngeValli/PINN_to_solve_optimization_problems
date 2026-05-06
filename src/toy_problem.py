"""
toy_problem.py
==============
Defines the toy constrained optimisation problem used throughout the tutorial:

    min  f(x, y) = x^2 + y^2
    s.t. h(x, y) = x + y - 1 = 0

Analytical solution: x* = y* = 0.5, lambda* = -1.

All functions are written with JAX so that they are fully differentiable
and can be JIT-compiled.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Problem functions
# ---------------------------------------------------------------------------

def objective(xy: jnp.ndarray) -> jnp.ndarray:
    """Objective function  f(x, y) = x^2 + y^2.

    Parameters
    ----------
    xy : array of shape (2,)
        Primal variables [x, y].

    Returns
    -------
    Scalar value of the objective.
    """
    x, y = xy[0], xy[1]
    return x ** 2 + y ** 2


def equality_constraint(xy: jnp.ndarray) -> jnp.ndarray:
    """Equality constraint  h(x, y) = x + y - 1 = 0.

    Parameters
    ----------
    xy : array of shape (2,)
        Primal variables [x, y].

    Returns
    -------
    Scalar residual of the equality constraint.
    """
    x, y = xy[0], xy[1]
    return x + y - 1.0


def lagrangian(xy: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """Lagrangian  L(x, y, lambda) = f(x, y) + lambda * h(x, y).

    For this problem there are no inequality constraints, so there is no
    mu term.

    Parameters
    ----------
    xy : array of shape (2,)
        Primal variables [x, y].
    lam : array of shape (1,) or scalar
        Lagrange multiplier for the equality constraint.

    Returns
    -------
    Scalar value of the Lagrangian.
    """
    return objective(xy) + lam[0] * equality_constraint(xy)


# ---------------------------------------------------------------------------
# Gradient-flow right-hand sides  (KKT → IVP)
# ---------------------------------------------------------------------------

def primal_rhs(xy: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """RHS for the primal variables: dx/dt = -∇_x L.

    Parameters
    ----------
    xy : array of shape (2,)
    lam : array of shape (1,)

    Returns
    -------
    Array of shape (2,).
    """
    grad_xy = jax.grad(lagrangian, argnums=0)(xy, lam)
    return -grad_xy


def dual_rhs(xy: jnp.ndarray) -> jnp.ndarray:
    """RHS for the dual variable: dlambda/dt = ∇_lambda L = h(x).

    Parameters
    ----------
    xy : array of shape (2,)

    Returns
    -------
    Array of shape (1,).
    """
    return jnp.array([equality_constraint(xy)])


# ---------------------------------------------------------------------------
# Analytical solution (for verification)
# ---------------------------------------------------------------------------

ANALYTICAL_SOLUTION = {
    "x": 0.5,
    "y": 0.5,
    "lambda": -1.0,
}
