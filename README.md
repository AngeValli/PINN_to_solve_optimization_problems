# PINN to Solve Optimization Problems

A step-by-step tutorial demonstrating how to solve constrained optimization problems by reformulating their **Karush-Kuhn-Tucker (KKT) conditions** as an **Initial Value Problem (IVP)** and solving it with **Physics-Informed Neural Networks (PINNs)** using [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax).

---

## Table of Contents

1. [Mathematical Background](#mathematical-background)
   - [Optimization Problem](#optimization-problem)
   - [KKT Conditions](#kkt-conditions)
   - [IVP Formulation via Primal-Dual Gradient Flow](#ivp-formulation-via-primal-dual-gradient-flow)
   - [PINN Approach](#pinn-approach)
2. [Toy Problem](#toy-problem)
3. [Repository Structure](#repository-structure)
4. [Installation and Usage](#installation-and-usage)
5. [Results](#results)

---

## Mathematical Background

### Optimization Problem

We consider the standard non-linear programming (NLP) problem:

$$\min_{x \in \mathbb{R}^n} \quad f(x)$$

$$\text{subject to} \quad h_i(x) = 0, \quad i = 1, \dots, p$$

$$\quad\quad\quad\quad\quad\;\; g_j(x) \le 0, \quad j = 1, \dots, m$$

where $f : \mathbb{R}^n \to \mathbb{R}$ is the objective, $h : \mathbb{R}^n \to \mathbb{R}^p$ are equality constraints, and $g : \mathbb{R}^n \to \mathbb{R}^m$ are inequality constraints.

The **Lagrangian** associated with this problem is:

$$L(x, \lambda, \mu) = f(x) + \lambda^\top h(x) + \mu^\top g(x)$$

where $\lambda \in \mathbb{R}^p$ are the equality multipliers and $\mu \in \mathbb{R}^m_{\ge 0}$ are the inequality (KKT) multipliers.

---

### KKT Conditions

A point $x^*$ is a local minimum (under suitable regularity conditions) if and only if there exist multipliers $\lambda^* \in \mathbb{R}^p$ and $\mu^* \in \mathbb{R}^m_{\ge 0}$ satisfying the **KKT conditions**:

1. **Stationarity**
$$\nabla_x L(x^*, \lambda^*, \mu^*) = \nabla f(x^*) + \sum_i \lambda^*_i \nabla h_i(x^*) + \sum_j \mu^*_j \nabla g_j(x^*) = 0$$

2. **Primal Feasibility**
$$h(x^*) = 0, \qquad g(x^*) \le 0$$

3. **Dual Feasibility**
$$\mu^* \ge 0$$

4. **Complementary Slackness**
$$\mu^*_j \, g_j(x^*) = 0, \quad j = 1, \dots, m$$

Finding $(x^*, \lambda^*, \mu^*)$ satisfying these conditions is equivalent to finding the root of a coupled system of equations.

---

### IVP Formulation via Primal-Dual Gradient Flow

Instead of solving the KKT system as a static root-finding problem, we **embed it in a continuous-time dynamical system** whose equilibrium coincides with the KKT point. This is achieved via **primal-dual gradient flow** dynamics:

$$\frac{dx}{dt} = -\nabla_x L(x, \lambda, \mu)$$

$$\frac{d\lambda}{dt} = \nabla_\lambda L(x, \lambda, \mu) = h(x)$$

$$\frac{d\mu}{dt} = \left[\nabla_\mu L(x, \lambda, \mu)\right]_+ = \left[g(x)\right]_+$$

where $[\cdot]_+ = \max(\cdot, 0)$ projects onto the non-negative orthant, enforcing dual feasibility $\mu \ge 0$. At the equilibrium $\frac{dz}{dt} = 0$ (where $z = (x, \lambda, \mu)$), the KKT conditions are recovered.

We pair this ODE system with **initial conditions** $z(0) = z_0$ (arbitrary starting point) to form a well-posed **Initial Value Problem (IVP)**:

$$\dot{z}(t) = F(z(t)), \qquad z(0) = z_0, \qquad t \in [0, T]$$

---

### PINN Approach

Rather than using a classical numerical integrator (e.g., Runge-Kutta), we parameterize the solution trajectories $z(t) = (x(t), \lambda(t), \mu(t))$ using a **neural network** $\hat{z}_\theta(t)$ with parameters $\theta$.

The network is trained to minimize a **physics-informed loss** composed of three terms:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{ODE}}(\theta) + \mathcal{L}_{\text{IC}}(\theta) + \mathcal{L}_{\text{CS}}(\theta)$$

| Term | Expression | Purpose |
|---|---|---|
| **ODE residual** | $\mathcal{L}_{\text{ODE}} = \frac{1}{N}\sum_{k=1}^{N}\left\|\dot{\hat{z}}_\theta(t_k) - F(\hat{z}_\theta(t_k))\right\|^2$ | Enforce gradient-flow dynamics |
| **Initial condition** | $\mathcal{L}_{\text{IC}} = \left\|\hat{z}_\theta(0) - z_0\right\|^2$ | Anchor the trajectory at $t=0$ |
| **Compl. slackness** | $\mathcal{L}_{\text{CS}} = \frac{1}{N}\sum_{k=1}^{N}\left\|\hat{\mu}_\theta(t_k) \odot g(\hat{x}_\theta(t_k))\right\|^2$ | Enforce $\mu \cdot g(x) = 0$ |

The time derivative $\dot{\hat{z}}_\theta(t_k)$ is computed **exactly** via `jax.grad`, making the loss fully differentiable and enabling gradient-based optimization with Optax.

---

## Toy Problem

The tutorial uses the following small but illustrative problem:

$$\min_{x, y} \quad f(x, y) = x^2 + y^2$$

$$\text{subject to} \quad h(x, y) = x + y - 1 = 0$$

The analytical solution is $x^* = y^* = 0.5$ with Lagrange multiplier $\lambda^* = -1$. The PINN should converge to this point from an arbitrary initialization.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── toy_problem.py    # Objective, constraints, and Lagrangian (JAX)
│   ├── pinn_solver.py    # Flax neural network, PINN loss, Optax training
│   └── main.py           # Orchestration: train, evaluate, plot
└── notebooks/
    └── tutorial.ipynb    # Standalone Colab-ready interactive tutorial
```

---

## Installation and Usage

This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible environment management. You can run the project in two different ways depending on your workflow and preferences. For a standard local execution, you can directly launch the `main.py` script, which orchestrates the full pipeline including training, evaluation, and visualization. Alternatively, if you prefer an interactive and cloud-based environment, you can use the Jupyter notebook provided in `notebooks/tutorial.ipynb`. The notebook is fully compatible with Google Colab, allowing you to run the tutorial without installing anything locally while still being able to modify parameters, inspect intermediate results, and experiment interactively.

## Using script file

### 1. Install uv (if not already installed)

```bash
pip install uv
# or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and activate the virtual environment

```bash
uv venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Run the example

```bash
python src/main.py
```

This trains the PINN, prints the loss at each epoch (every 500 steps), and opens two interactive **Plotly** charts showing the convergence of $(x(t), y(t), \lambda(t))$ toward the KKT point and the training loss curve.

## Using notebook

## 1. Interactive notebook (Google Colab)

Open `notebooks/tutorial.ipynb` directly in [Google Colab](https://colab.research.google.com/) — the first cell installs all required packages automatically via `pip` (the standard Colab way). For local execution, use `uv pip install -r requirements.txt` as above.

---

## Results

After training, the PINN trajectories converge to the analytical solution:

| Variable | PINN result | Analytical |
|---|---|---|
| $x^*$ | ≈ 0.50 | 0.50 |
| $y^*$ | ≈ 0.50 | 0.50 |
| $\lambda^*$ | ≈ −1.00 | −1.00 |

The interactive **Plotly** charts produced by `src/main.py` visualise this convergence over the pseudo-time domain $t \in [0, T]$.

---

## References

* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics, 378, 686-707.
* Arrow, K. J., Hurwicz, L., Uzawa, H., Chenery, H. B., Johnson, S., & Karlin, S. (1958). Studies in linear and non-linear programming (Vol. 2). Stanford: Stanford University Press.
* Karush, W. (1939). Minima of functions of several variables with inequalities as side constraints. M. Sc. Dissertation. Dept. of Mathematics, Univ. of Chicago.
* Kuhn, H. W., & Tucker, A. W. (1951). Proceedings of 2nd berkeley symposium. In Proc. 2nd Berkeley Symp. (pp. 481-492).