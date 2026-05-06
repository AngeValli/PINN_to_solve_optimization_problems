"""
main.py
=======
Orchestrates the PINN training for the toy optimisation problem, evaluates
the learned trajectories, prints a comparison with the analytical solution,
and produces interactive Plotly charts for the loss curve and the
primal/dual variable trajectories.

Usage
-----
    python src/main.py
"""

import os
import sys

# Allow imports from the same directory when called as a script
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from toy_problem import ANALYTICAL_SOLUTION
from pinn_solver import PINN, train


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

T = 5.0          # pseudo-time horizon for the gradient-flow IVP
N_COLLOC = 100   # number of collocation points used during training
N_EPOCHS = 5000  # total gradient-descent steps
LR = 1e-3        # Adam learning rate
SEED = 42        # PRNG seed for reproducibility

# Initial conditions  z(0) = [x0, y0, lambda0]
# The PINN must learn to evolve from this arbitrary starting point to the
# KKT solution (x*=0.5, y*=0.5, lambda*=-1).
Z0 = jnp.array([0.0, 2.0, 0.0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Train the PINN, evaluate trajectories, and produce interactive plots.

    Steps:
        1. Instantiate the Flax PINN model.
        2. Run the Optax training loop (see pinn_solver.train).
        3. Evaluate the trained network on a fine time grid.
        4. Print a numeric comparison with the analytical solution.
        5. Display interactive Plotly figures for the loss curve and
           the primal/dual trajectories.
    """
    print("=" * 60)
    print("PINN solver for KKT / IVP toy problem")
    print("=" * 60)
    print("Problem : min x^2 + y^2  s.t.  x + y = 1")
    print(f"Analytical solution: {ANALYTICAL_SOLUTION}")
    print(
        f"Initial conditions : x0={float(Z0[0])}, y0={float(Z0[1])}, "
        f"lambda0={float(Z0[2])}"
    )
    print("-" * 60)

    # ------------------------------------------------------------------
    # 1. Build and train the model
    # ------------------------------------------------------------------
    model = PINN(hidden_sizes=(32, 32, 32))

    params, loss_history = train(
        model,
        z0=Z0,
        T=T,
        n_colloc=N_COLLOC,
        n_epochs=N_EPOCHS,
        lr=LR,
        seed=SEED,
        print_every=500,
    )

    # ------------------------------------------------------------------
    # 2. Evaluate on a fine time grid
    #    jax.vmap vectorises the network call over all time points at once,
    #    which is much faster than a Python for-loop.
    # ------------------------------------------------------------------
    t_eval = jnp.linspace(0.0, T, 300)
    trajectories = jax.vmap(
        lambda t: model.apply({"params": params}, t)
    )(t_eval)   # shape (300, 3)

    x_traj   = trajectories[:, 0]   # primal variable x(t)
    y_traj   = trajectories[:, 1]   # primal variable y(t)
    lam_traj = trajectories[:, 2]   # dual variable lambda(t)

    # ------------------------------------------------------------------
    # 3. Print numeric comparison
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Final PINN values (at t = T):")
    print(f"  x       = {float(x_traj[-1]):.6f}  (analytical: {ANALYTICAL_SOLUTION['x']})")
    print(f"  y       = {float(y_traj[-1]):.6f}  (analytical: {ANALYTICAL_SOLUTION['y']})")
    print(f"  lambda  = {float(lam_traj[-1]):.6f}  (analytical: {ANALYTICAL_SOLUTION['lambda']})")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 4. Interactive Plotly: trajectory convergence (2-row layout)
    # ------------------------------------------------------------------
    t_np = [float(v) for v in t_eval]   # plain Python list for Plotly

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Primal variable x", "Primal variable y", "Dual variable λ"),
    )
    fig.update_layout(title_text="PINN Trajectories Converging to the KKT Point")

    # Primal x
    fig.add_trace(
        go.Scatter(x=t_np, y=[float(v) for v in x_traj],
                   mode="lines", name="x̂(t) PINN",
                   line=dict(color="steelblue")),
        row=1, col=1,
    )
    fig.add_hline(
        y=ANALYTICAL_SOLUTION["x"], line_dash="dash", line_color="steelblue",
        annotation_text="x*=0.5", row=1, col=1,
    )

    # Primal y
    fig.add_trace(
        go.Scatter(x=t_np, y=[float(v) for v in y_traj],
                   mode="lines", name="ŷ(t) PINN",
                   line=dict(color="darkorange")),
        row=1, col=2,
    )
    fig.add_hline(
        y=ANALYTICAL_SOLUTION["y"], line_dash="dash", line_color="darkorange",
        annotation_text="y*=0.5", row=1, col=2,
    )

    # Dual lambda
    fig.add_trace(
        go.Scatter(x=t_np, y=[float(v) for v in lam_traj],
                   mode="lines", name="λ̂(t) PINN",
                   line=dict(color="seagreen")),
        row=1, col=3,
    )
    fig.add_hline(
        y=ANALYTICAL_SOLUTION["lambda"], line_dash="dash", line_color="seagreen",
        annotation_text="λ*=−1", row=1, col=3,
    )

    fig.update_xaxes(title_text="t")
    fig.show()

    # ------------------------------------------------------------------
    # 5. Interactive Plotly: training loss curve (log scale)
    # ------------------------------------------------------------------
    epochs = list(range(1, len(loss_history) + 1))
    fig_loss = go.Figure(
        go.Scatter(x=epochs, y=loss_history,
                   mode="lines", name="Loss",
                   line=dict(color="crimson"))
    )
    fig_loss.update_layout(
        title="PINN Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",       # log scale makes convergence visible
    )
    fig_loss.show()


if __name__ == "__main__":
    main()

