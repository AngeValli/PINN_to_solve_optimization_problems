"""
pinn_solver.py
==============
Implements the Physics-Informed Neural Network (PINN) that solves the
primal-dual gradient-flow IVP for the toy optimisation problem.

Architecture
------------
The network maps a scalar time t ∈ [0, T] to the state vector

    z(t) = [x(t), y(t), lambda(t)]

i.e. 1 input neuron → hidden layers with tanh activations → 3 output neurons.

Loss function
-------------
    L_total = w_ode * L_ode + w_ic * L_ic

where
    L_ode : mean-squared ODE residual at collocation points
            — enforces the gradient-flow dynamics dz/dt = F(z)
    L_ic  : squared error at t = 0
            — anchors the trajectory to the given initial conditions

Key JAX concepts used
---------------------
* ``jax.grad`` / ``jax.jacfwd``
    Exact automatic differentiation (AD) — no finite differences.
    ``jacfwd`` uses *forward-mode* AD to compute all output derivatives w.r.t.
    a scalar input in one pass, making it efficient for our 1-input / 3-output
    network.

* ``jax.vmap``
    Vectorises a function over a batch axis without explicit Python loops.
    We use it to evaluate the ODE residual at all N collocation points at once.

* ``jax.jit``
    Just-In-Time compilation via XLA.  The compiled function runs on the
    accelerator (CPU/GPU/TPU) with near-native speed and avoids Python
    overhead on every training step.

* ``jax.value_and_grad``
    Returns both the loss value and its gradient w.r.t. the first argument
    in a single forward+backward pass — exactly what the Optax update step
    needs.
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
    """Fully-connected PINN: scalar time t → state vector z(t) = [x, y, λ].

    The network is defined as a Flax ``Module``, which means:
      * All learnable parameters (Dense weights and biases) are stored
        *outside* the module in a pytree (``params``), not inside the object.
      * ``@nn.compact`` lets us declare sub-modules (``nn.Dense``) inline
        inside ``__call__``, rather than in a separate ``setup`` method.
      * ``model.apply({"params": params}, t)`` performs a pure functional
        forward pass — no hidden mutable state, so it is safe to JIT/vmap/grad
        through.

    Attributes:
        hidden_sizes: Sequence of integers specifying the number of neurons
            in each hidden layer, e.g. ``(32, 32, 32)`` for three layers of
            32 neurons.
    """

    hidden_sizes: Sequence[int] = (32, 32, 32)

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: map a scalar time value to the 3-D state vector.

        Args:
            t: Scalar time value (or shape-``()`` array).  Will be promoted
                to shape ``(1,)`` internally to feed the first Dense layer.

        Returns:
            Array of shape ``(3,)`` containing [x(t), y(t), λ(t)].
        """
        # Promote scalar t to a 1-D vector so nn.Dense can process it
        z = jnp.atleast_1d(t)                  # shape: (1,)

        # Stack of hidden layers with tanh activations.
        # tanh is preferred over ReLU for PINNs because:
        #   1. It is smooth (infinitely differentiable), which matters when we
        #      differentiate the network w.r.t. t to get dz/dt.
        #   2. It is bounded, which helps prevent gradient explosion.
        for size in self.hidden_sizes:
            z = nn.Dense(size)(z)               # affine: W z + b
            z = nn.tanh(z)                      # element-wise non-linearity

        z = nn.Dense(3)(z)                      # linear readout → [x, y, λ]
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
    """Computes the total physics-informed loss for a single parameter set.

    The loss has two terms:

    1. **ODE residual** (L_ode):
       At each collocation point t_k, we compute the network's time derivative
       dẑ/dt (via forward-mode AD) and compare it to the gradient-flow RHS
       F(ẑ(t_k)).  The loss penalises any mismatch, driving the network to
       satisfy the ODE.

    2. **Initial condition** (L_ic):
       The network output at t = 0 should match z0.  The higher default weight
       ``w_ic = 10`` prioritises satisfying the IC over the ODE residual,
       which helps the optimiser find a unique trajectory.

    Args:
        params: Flax parameter pytree (weights and biases of the network).
        apply_fn: Callable that performs a forward pass; typically
            ``model.apply``.  Signature: ``apply_fn({"params": params}, t)``.
        t_colloc: 1-D array of shape ``(N,)`` with the collocation time points
            in ``[0, T]``.
        z0: 1-D array of shape ``(3,)`` with the initial state
            ``[x0, y0, lambda0]``.
        w_ode: Scalar weight for the ODE residual term (default ``1.0``).
        w_ic: Scalar weight for the initial-condition term (default ``10.0``).

    Returns:
        A scalar JAX array representing the weighted sum of loss terms.
    """

    # -- Helper: evaluate network at a single time point --------------------
    # Closing over `params` makes `net` a pure function of t alone, which
    # is required by jax.jacfwd and jax.vmap.
    def net(t):
        return apply_fn({"params": params}, t)

    # -- Time derivative via *forward-mode* automatic differentiation -------
    # ``jax.jacfwd(net)`` returns a function that, given a scalar t, computes
    # the full Jacobian dnet(t)/dt of shape (3, 1) — or (3,) after squeeze.
    # Forward-mode AD is efficient here because the input is scalar (dim=1)
    # while the output has dim=3: one forward pass suffices.
    dnet_dt_vec = jax.jacfwd(net)

    # -- ODE residual at a single collocation point -------------------------
    def ode_residual(t):
        """Returns (dẑ/dt - F(ẑ)) at time t; shape (3,)."""
        z  = net(t)                         # [x, y, λ] — network prediction
        dz = dnet_dt_vec(t).squeeze()       # dz/dt via AD, shape (3,)

        # Split state into primal (xy) and dual (lam) components
        xy  = z[:2]                         # [x, y]
        lam = z[2:3]                        # [λ]

        # Evaluate gradient-flow RHS from toy_problem.py
        rhs_xy  = primal_rhs(xy, lam)      # -∇_x L, shape (2,)
        rhs_lam = dual_rhs(xy)             # h(x),   shape (1,)
        rhs = jnp.concatenate([rhs_xy, rhs_lam])   # shape (3,)

        return dz - rhs                    # zero iff ODE is satisfied

    # -- Vectorise over all collocation points with jax.vmap ----------------
    # ``jax.vmap(ode_residual)`` maps ode_residual over the leading axis of
    # t_colloc, returning an array of shape (N, 3) — one residual per point.
    # This is equivalent to a for-loop but is compiled into a single XLA op.
    residuals = jax.vmap(ode_residual)(t_colloc)   # shape (N, 3)
    L_ode = jnp.mean(residuals ** 2)               # mean squared residual

    # -- Initial-condition loss ---------------------------------------------
    # Evaluate the network at t = 0 and penalise deviation from z0
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
    """Trains the PINN using the Adam optimiser and returns the result.

    Training overview:
        1. Initialise the network parameters randomly using a JAX PRNG key.
        2. Build an Optax Adam optimiser and initialise its state.
        3. JIT-compile ``value_and_grad(pinn_loss)`` for speed.
        4. Run the gradient-descent loop: compute loss + grads, apply Optax
           update, repeat.

    Args:
        model: Uninitialised Flax ``PINN`` module.
        z0: Initial state array of shape ``(3,)`` — ``[x0, y0, lambda0]``.
        T: End of the pseudo-time horizon (default ``5.0``).
        n_colloc: Number of uniformly-spaced collocation points in
            ``[0, T]`` (default ``100``).
        n_epochs: Total number of gradient-descent steps (default ``5000``).
        lr: Adam learning rate (default ``1e-3``).
        seed: Integer seed for the JAX PRNG used to initialise weights
            (default ``0``).
        print_every: Print the current loss every this many epochs
            (default ``500``).

    Returns:
        A tuple ``(params, loss_history)`` where:
            params: Trained Flax parameter pytree.
            loss_history: Python list of scalar loss values, one per epoch.
    """
    # JAX requires explicit PRNG keys — there is no global random state.
    # PRNGKey(seed) creates a deterministic key; passing different seeds
    # gives different random initialisations.
    key = jax.random.PRNGKey(seed)

    # Uniformly-spaced collocation points covering the full time domain
    t_colloc = jnp.linspace(0.0, T, n_colloc)

    # Initialise network parameters.
    # ``model.init(key, dummy_input)`` does a single forward pass to infer
    # layer shapes, then initialises all Dense weights/biases randomly.
    # The result is a nested dict (pytree); we extract the "params" sub-tree.
    params = model.init(key, jnp.array(0.0))["params"]

    # Optax Adam: a popular adaptive gradient optimiser.
    # ``optimizer.init(params)`` creates the optimiser state (momentum buffers
    # etc.) that mirrors the structure of `params`.
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # ``jax.value_and_grad`` wraps pinn_loss so that a single call returns
    # both the scalar loss value and the gradient ∂L/∂params.
    # ``partial(...)`` pins apply_fn, t_colloc, and z0, leaving only `params`
    # as the differentiated argument.
    # ``jax.jit`` compiles the entire forward+backward pass to XLA, removing
    # Python interpreter overhead on every training step.
    loss_and_grad = jax.jit(
        jax.value_and_grad(
            partial(pinn_loss, apply_fn=model.apply, t_colloc=t_colloc, z0=z0)
        )
    )

    loss_history = []

    for epoch in range(1, n_epochs + 1):
        # Forward pass: compute loss; backward pass: compute gradients
        loss_val, grads = loss_and_grad(params)

        # Optax computes parameter updates from gradients and optimiser state
        updates, opt_state = optimizer.update(grads, opt_state)

        # Apply updates: params ← params + updates  (Adam-scaled step)
        params = optax.apply_updates(params, updates)

        loss_history.append(float(loss_val))

        if epoch % print_every == 0:
            print(f"Epoch {epoch:5d} | Loss {loss_val:.6e}")

    return params, loss_history

