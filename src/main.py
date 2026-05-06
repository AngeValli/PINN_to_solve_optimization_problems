"""
main.py
=======
Orchestrates the PINN training for the toy optimisation problem, evaluates
the learned trajectories, prints a comparison with the analytical solution,
and saves a plot to ``trajectories.png``.

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
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt

from toy_problem import ANALYTICAL_SOLUTION
from pinn_solver import PINN, train


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

T = 5.0          # time horizon
N_COLLOC = 100   # collocation points during training
N_EPOCHS = 5000  # training epochs
LR = 1e-3        # Adam learning rate
SEED = 42

# Initial conditions  z(0) = [x0, y0, lambda0]
Z0 = jnp.array([0.0, 2.0, 0.0])

OUTPUT_FILE = "trajectories.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PINN solver for KKT / IVP toy problem")
    print("=" * 60)
    print(f"Problem : min x^2 + y^2  s.t.  x + y = 1")
    print(f"Analytical solution: {ANALYTICAL_SOLUTION}")
    print(f"Initial conditions : x0={float(Z0[0])}, y0={float(Z0[1])}, "
          f"lambda0={float(Z0[2])}")
    print("-" * 60)

    # Build model
    model = PINN(hidden_sizes=(32, 32, 32))

    # Train
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

    # Evaluate on a fine grid
    t_eval = jnp.linspace(0.0, T, 300)
    trajectories = jax.vmap(
        lambda t: model.apply({"params": params}, t)
    )(t_eval)   # shape (300, 3)

    x_traj   = trajectories[:, 0]
    y_traj   = trajectories[:, 1]
    lam_traj = trajectories[:, 2]

    # Print final values
    print("-" * 60)
    print("Final PINN values (at t = T):")
    print(f"  x       = {float(x_traj[-1]):.6f}  (analytical: {ANALYTICAL_SOLUTION['x']})")
    print(f"  y       = {float(y_traj[-1]):.6f}  (analytical: {ANALYTICAL_SOLUTION['y']})")
    print(f"  lambda  = {float(lam_traj[-1]):.6f}  (analytical: {ANALYTICAL_SOLUTION['lambda']})")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PINN Trajectories Converging to the KKT Point", fontsize=13)

    t_np = t_eval

    # Primal variable x
    axes[0].plot(t_np, x_traj, label=r"$\hat{x}(t)$ PINN", color="steelblue")
    axes[0].axhline(ANALYTICAL_SOLUTION["x"], color="steelblue",
                    linestyle="--", label=r"$x^*=0.5$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x(t)")
    axes[0].set_title("Primal variable x")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Primal variable y
    axes[1].plot(t_np, y_traj, label=r"$\hat{y}(t)$ PINN", color="darkorange")
    axes[1].axhline(ANALYTICAL_SOLUTION["y"], color="darkorange",
                    linestyle="--", label=r"$y^*=0.5$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("y(t)")
    axes[1].set_title("Primal variable y")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Dual variable lambda
    axes[2].plot(t_np, lam_traj, label=r"$\hat{\lambda}(t)$ PINN", color="seagreen")
    axes[2].axhline(ANALYTICAL_SOLUTION["lambda"], color="seagreen",
                    linestyle="--", label=r"$\lambda^*=-1$")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel(r"$\lambda(t)$")
    axes[2].set_title("Dual variable λ")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"\nPlot saved to: {OUTPUT_FILE}")

    # Also save loss curve
    loss_file = "loss_curve.png"
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.semilogy(loss_history, color="crimson")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("PINN Training Loss")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(loss_file, dpi=150)
    print(f"Loss curve saved to: {loss_file}")


if __name__ == "__main__":
    main()
