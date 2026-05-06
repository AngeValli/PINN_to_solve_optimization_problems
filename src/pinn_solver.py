"""
pinn_solver.py
==============
Implements the Physics-Informed Neural Network (PINN) that solves the
primal-dual gradient-flow IVP for the toy optimisation problem.

Architecture
------------
The network maps a scalar time t ∈ [0, T] to the state vector
    z(t) = [x(t), y(t), lambda(t)]
i.e. it has 1 input neuron and 3 output neurons.

Loss function
-------------
    L_total = w_ode * L_ode + w_ic * L_ic

where
    L_ode  : mean-squared ODE residual at collocation points
    L_ic   : squared error at t = 0 (initial conditions)

Time derivatives dz/dt are computed via jax.grad so that they are exact
(up to floating-point precision) and fully differentiable.
"""

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from toy_problem import primal_rhs, dual_rhs


# ---------------------------------------------------------------------------
# Neural-network architecture
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """Fully-connected network:  t  →  [x(t), y(t), lambda(t)].

    Attributes
    ----------
    hidden_sizes : sequence of ints
        Number of neurons in each hidden layer.
    """

    hidden_sizes: Sequence[int] = (32, 32, 32)

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Parameters
        ----------
        t : scalar or array of shape ()
            Normalised time value.

        Returns
        -------
        Array of shape (3,) — [x, y, lambda].
        """
        z = jnp.atleast_1d(t)          # shape (1,)
        for size in self.hidden_sizes:
            z = nn.Dense(size)(z)
            z = nn.tanh(z)
        z = nn.Dense(3)(z)             # output: [x, y, lambda]
        return z


# ---------------------------------------------------------------------------
# PINN loss
# ---------------------------------------------------------------------------

def pinn_loss(
    params,
    apply_fn,
    t_colloc: jnp.ndarray,
    z0: jnp.ndarray,
    w_ode: float = 1.0,
    w_ic: float = 10.0,
) -> jnp.ndarray:
    """Compute the total PINN loss.

    Parameters
    ----------
    params :
        Flax parameter tree.
    apply_fn :
        ``model.apply`` bound method (or equivalent callable).
    t_colloc : array of shape (N,)
        Collocation points in [0, T].
    z0 : array of shape (3,)
        Initial condition  z(0) = [x0, y0, lambda0].
    w_ode : float
        Weight for the ODE residual term.
    w_ic : float
        Weight for the initial-condition term.

    Returns
    -------
    Scalar loss value.
    """

    # -- helper: evaluate network at a single time point --------------------
    def net(t):
        return apply_fn({"params": params}, t)

    # -- time derivative via automatic differentiation ----------------------
    # jacfwd gives the full Jacobian dz/dt as a vector of shape (3,)
    dnet_dt_vec = jax.jacfwd(net)

    # -- ODE residual at collocation points ---------------------------------
    def ode_residual(t):
        z = net(t)                        # [x, y, lambda]
        dz = dnet_dt_vec(t).squeeze()     # shape (3,)

        xy = z[:2]
        lam = z[2:3]

        rhs_xy = primal_rhs(xy, lam)     # shape (2,)
        rhs_lam = dual_rhs(xy)           # shape (1,)
        rhs = jnp.concatenate([rhs_xy, rhs_lam])  # shape (3,)

        return dz - rhs                  # residual; 0 when ODE is satisfied

    # vmap over collocation points
    residuals = jax.vmap(ode_residual)(t_colloc)   # (N, 3)
    L_ode = jnp.mean(residuals ** 2)

    # -- initial-condition loss ---------------------------------------------
    z_pred_0 = net(jnp.array(0.0))
    L_ic = jnp.mean((z_pred_0 - z0) ** 2)

    return w_ode * L_ode + w_ic * L_ic


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: PINN,
    z0: jnp.ndarray,
    T: float = 5.0,
    n_colloc: int = 100,
    n_epochs: int = 5000,
    lr: float = 1e-3,
    seed: int = 0,
    print_every: int = 500,
):
    """Train the PINN and return the optimised parameters + loss history.

    Parameters
    ----------
    model : PINN
        Flax module (uninitialised).
    z0 : array of shape (3,)
        Initial conditions [x0, y0, lambda0].
    T : float
        End of the time horizon.
    n_colloc : int
        Number of uniformly-spaced collocation points in [0, T].
    n_epochs : int
        Number of gradient-descent steps.
    lr : float
        Adam learning rate.
    seed : int
        PRNG seed for weight initialisation.
    print_every : int
        Print loss every this many epochs.

    Returns
    -------
    params : parameter tree
        Trained network parameters.
    loss_history : list of floats
        Loss value recorded at every epoch.
    """
    key = jax.random.PRNGKey(seed)
    t_colloc = jnp.linspace(0.0, T, n_colloc)

    # Initialise parameters
    params = model.init(key, jnp.array(0.0))["params"]

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # JIT-compile loss + grad
    loss_and_grad = jax.jit(
        jax.value_and_grad(
            partial(pinn_loss, apply_fn=model.apply, t_colloc=t_colloc, z0=z0)
        )
    )

    loss_history = []

    for epoch in range(1, n_epochs + 1):
        loss_val, grads = loss_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        loss_history.append(float(loss_val))

        if epoch % print_every == 0:
            print(f"Epoch {epoch:5d} | Loss {loss_val:.6e}")

    return params, loss_history
