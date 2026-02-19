# NCP: Nonparametric Chain Policies

A PyTorch implementation of **Nonparametric Chain Policies (NCP)** for data-driven practical stabilization of nonlinear systems.

NCP constructs a certified feedback controller by adaptively partitioning the state space into hypercube cells, finding MPPI-based control sequences that contract each cell toward the origin, and verifying contraction rates via binary search over a Lyapunov-like certificate. The result is a lookup-table controller with formal stability guarantees.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+ and PyTorch 2.0+.

## Quick Start

```python
import torch
import ncp

# Build an NCP for the inverted pendulum
result = ncp.NCPBuilder(
    env_factory=ncp.PendulumEnv,
    state_spec=ncp.StateSpec(dim=2, wrap_dims=(0,)),
    control_spec=ncp.ControlSpec(
        dim=1,
        lower_bounds=torch.tensor([-2.0]),
        upper_bounds=torch.tensor([2.0]),
    ),
    dynamics_fn=ncp.inverted_pendulum_2d_torch,
    radius=2.0,
    epsilon=0.1,
    lipschitz=5.0,
    tau=0.5,
    dt=0.1,
    min_alpha=0.01,
    max_splits=3,
    num_samples=500,
).build()

print(f"Verified {result.verified_centers.shape[0]} cells in {result.elapsed_seconds:.1f}s")

# Use the lookup controller
controller = ncp.NCPController(result)
ctrl, tau = controller.lookup(torch.tensor([[0.5, 0.0]]))
```

## Repository Structure

```
ncp/
  envs/           # Environment abstraction (BaseEnv + system-specific subclasses)
  dynamics/        # Continuous-time dynamics functions (15 systems)
  geometry/        # Hypercube operations, intersection queries, grid generation
  algorithm/       # Core NCP: MPPI search, alpha binary search, builder loop
  evaluation/      # Lookup controller and closed-loop simulation
  visualization/   # Cell and trajectory plotting utilities
  utils/           # Angle wrapping, tensor operations

experiments/
  ncp_vs_mpc/      # NCP vs MPC comparison across all 15 systems
    src/            # Runner scripts, controllers, analysis code
    configs/        # Per-system and per-phase configuration JSONs
    slurm/          # SLURM array job scripts for cluster execution
    results/        # Metrics JSONs from completed runs

tests/             # 158 tests covering all modules
```

### Key Components

| Class / Function | Description |
|---|---|
| `NCPBuilder` | System-agnostic NCP algorithm driver. Handles adaptive grid refinement, MPPI search, and contraction-rate verification. Supports periodic checkpointing via `checkpoint_fn` callback. |
| `NCPResult` | Dataclass storing verified cells, control sequences, contraction rates, and timing. |
| `NCPController` | Lookup-table controller that maps states to stored control sequences. |
| `BaseEnv` | Abstract environment with trajectory rollout, MPPI sampling, and distance computation. Subclass to add new systems. |
| `search_alpha_parallel` | Batched binary search for the tightest contraction rate per cell. |
| `simulate` | Closed-loop simulation of the NCP controller from initial conditions. |

### Included Dynamical Systems (15)

| System | State | Control | Wrap dims | Field |
|---|---|---|---|---|
| **Inverted Pendulum** | 2D `[theta, theta_dot]` | 1 (torque) | `(0,)` | Classic control |
| **Unicycle** | 3D `[x, y, theta]` | 2 `[v, omega]` | `(2,)` | Mobile robotics |
| **Bicycle** | 3D `[x, y, theta]` | 2 `[v, delta]` | `(2,)` | Mobile robotics |
| **Cart-Pole** | 4D `[x, x_dot, theta, theta_dot]` | 1 (force) | `()` | RL / LQR / MPC |
| **Acrobot** | 4D `[q1, q2, q1_dot, q2_dot]` | 1 (tau2) | `(0, 1)` | RL / underactuated |
| **Mountain Car** | 2D `[position, velocity]` | 1 (force) | `()` | RL |
| **Pendubot** | 4D `[q1, q2, q1_dot, q2_dot]` | 1 (tau1) | `(0, 1)` | Underactuated |
| **Furuta Pendulum** | 4D `[theta_arm, theta_pend, ...]` | 1 (torque) | `(0, 1)` | Control labs |
| **Ball and Beam** | 4D `[r, r_dot, theta, theta_dot]` | 1 (torque) | `()` | MPC |
| **Planar Quadrotor** | 6D `[x, y, x_dot, y_dot, phi, phi_dot]` | 2 `[f1, f2]` | `(4,)` | Aerospace / NMPC |
| **2-Link Robot Arm** | 4D `[q1, q2, q1_dot, q2_dot]` | 2 `[tau1, tau2]` | `(0, 1)` | Robotics |
| **CSTR** | 2D `[C_A, T]` | 1 (heat Q) | `()` | MPC / process |
| **Van der Pol** | 2D `[x, x_dot]` | 1 (force) | `()` | Nonlinear control |
| **Duffing Oscillator** | 2D `[x, x_dot]` | 1 (force) | `()` | Chaos control |
| **Lotka-Volterra** | 2D `[prey, predator]` | 1 (harvest) | `()` | Ecological / bio |

## Experiments: NCP vs MPC

The `experiments/ncp_vs_mpc/` directory contains a full comparison of NCP against two MPC baselines across all 15 dynamical systems.

### Three Controllers

| Controller | Method | Hardware | Approach |
|---|---|---|---|
| **NCP** | Offline lookup-table with contraction guarantees | GPU | One-time build, then instant lookup |
| **MPPI-MPC** | Online sampling-based MPC (PyTorch) | GPU | Re-plan every step via MPPI sampling |
| **CasADi/IPOPT NMPC** | Online optimization-based MPC | CPU | Multiple-shooting with IPOPT solver |

### Six Evaluation Modes

| Mode | Description |
|---|---|
| `ncp_native` | Apply full NCP control sequence (up to tau steps), then re-query |
| `ncp_step` | Apply one NCP control per step, re-query every step |
| `ncp_nn_native` | Like native, but nearest-neighbor fallback outside verified region |
| `ncp_nn_step` | Like step, but nearest-neighbor fallback outside verified region |
| `mppi_mpc` | MPPI-MPC re-plans at every step |
| `casadi_mpc` | CasADi NMPC re-plans at every step |

### Experiment Files

```
experiments/ncp_vs_mpc/
  src/
    registry.py           # System name -> env class, dynamics, specs
    config.py             # JSON config loader with phase merge
    initial_conditions.py # Grid + random IC generation
    metrics.py            # NCP/MPC/ClosedLoop metric dataclasses
    mppi_mpc.py           # MPPI-MPC controller (PyTorch, GPU)
    casadi_mpc.py         # CasADi/IPOPT NMPC controller (CPU)
    casadi_dynamics/      # 15 CasADi symbolic dynamics translations
    run_ncp.py            # CLI: build NCP for one system (with checkpointing)
    run_mppi_mpc.py       # CLI: run MPPI-MPC for one system
    run_casadi_mpc.py     # CLI: run CasADi-MPC for one system
    run_closed_loop.py    # CLI: evaluate all controllers on shared ICs
    aggregate_results.py  # Collect into summary CSV/JSON
    plot_results.py       # Comparison figures
  configs/
    systems/              # 15 base configs (one per system)
    phase1/               # Small validation configs
    phase2/               # Production-scale configs
  slurm/                  # SLURM array job scripts (0-14 for 15 systems)
  results/                # Metrics JSONs from completed runs
  alpha_analysis.py       # Compute exponential convergence rates
```

### Running the Experiments

**Prerequisites**: PyTorch 2.0+, CasADi (`pip install casadi`), access to a SLURM cluster with GPUs.

```bash
# Single system locally
cd experiments/ncp_vs_mpc
python src/run_ncp.py --system pendulum --phase 1
python src/run_mppi_mpc.py --system pendulum --phase 1
python src/run_casadi_mpc.py --system pendulum --phase 1
python src/run_closed_loop.py --system pendulum --phase 1

# All 15 systems via SLURM (array jobs)
bash slurm/submit_phase1.sh
```

### Metrics Collected

| Category | Metric |
|---|---|
| **Timing** | Build time (NCP), per-step solve time, total wall time |
| **NCP-specific** | Verified cells, coverage ratio, contraction rate (alpha) |
| **Closed-loop** | Convergence rate, steps to converge, trajectory cost, final error, control energy |
| **Convergence analysis** | Exponential decay rate via least-squares fit on log(||x(t)||) vs t |

## Adding a New System

1. Write a dynamics function `(x, u, **kwargs) -> dx/dt` operating on batched tensors.
2. Create a thin `BaseEnv` subclass specifying `StateSpec`, `ControlSpec`, and the dynamics.
3. Pass it to `NCPBuilder`.

```python
def my_dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # x: (B, state_dim), u: (B, ctrl_dim) -> dx/dt: (B, state_dim)
    return -0.5 * x + u

result = ncp.NCPBuilder(
    env_factory=lambda **kw: ncp.BaseEnv(
        state_spec=ncp.StateSpec(dim=4),
        control_spec=ncp.ControlSpec(dim=4, lower_bounds=-torch.ones(4), upper_bounds=torch.ones(4)),
        dynamics_fn=my_dynamics,
        **kw,
    ),
    state_spec=ncp.StateSpec(dim=4),
    control_spec=ncp.ControlSpec(dim=4, lower_bounds=-torch.ones(4), upper_bounds=torch.ones(4)),
    dynamics_fn=my_dynamics,
    radius=1.0, epsilon=0.1, lipschitz=1.0, tau=1.0, dt=0.1,
    min_alpha=0.001, max_splits=2, num_samples=200,
    angle_bound_dims=(),
).build()
```

## Algorithm Overview

1. **Grid initialization**: Cover `[-R, R]^d` with multi-resolution hypercube cells.
2. **MPPI search**: For each batch of cells, sample control trajectories and select the one with the best contraction metric.
3. **Alpha verification**: Binary-search for the tightest contraction rate `alpha` satisfying `min_t [||phi(t)|| * exp(alpha*t) + r * exp((alpha+L)*t)] < V(x) - r`.
4. **Subdivision**: Cells that fail verification are split into `3^d` sub-cells and re-processed.
5. **Bootstrapping** (optional): Trajectories that enter previously verified regions can inherit their contraction guarantees.
6. **Checkpointing** (optional): Pass a `checkpoint_fn` callback to `build()` to save partial results periodically, protecting against timeouts.

## Parameters

| Parameter | Description | Typical range |
|---|---|---|
| `radius` | Outer radius of the state-space region to cover | Problem-dependent |
| `epsilon` | Finest cell half-width (resolution) | 0.01 -- 0.5 |
| `lipschitz` | Lipschitz constant of the dynamics | System-dependent |
| `tau` | Trajectory time horizon | 0.5 -- 5.0 |
| `dt` | Euler integration step | 0.01 -- 0.1 |
| `min_alpha` | Minimum acceptable contraction rate | 0.0001 -- 0.01 |
| `max_splits` | Maximum subdivision depth | 2 -- 5 |
| `num_samples` | MPPI samples per iteration | 200 -- 2000 |
| `reuse` | Enable bootstrapping from verified cells | `True` for large problems |
| `checkpoint_fn` | Callback for periodic saves during `build()` | `None` or `Callable` |
| `checkpoint_interval` | Seconds between checkpoint calls | 300 (5 min) |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

158 tests covering dynamics, geometry, environments, algorithm modules, and full end-to-end pipelines across all 15 systems.

## License

MIT
