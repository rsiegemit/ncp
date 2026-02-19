"""Metric dataclasses for NCP, MPC, and closed-loop evaluation."""

import json
import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class NCPMetrics:
    system_name: str = ""
    build_time_seconds: float = 0.0
    num_verified_cells: int = 0
    num_unverified_cells: int = 0
    coverage_ratio: float = 0.0
    alpha_mean: float = 0.0
    alpha_median: float = 0.0
    alpha_min: float = 0.0
    alpha_max: float = 0.0
    # Per-step lookup time (from closed-loop eval)
    lookup_time_mean: float = 0.0
    lookup_time_median: float = 0.0
    lookup_time_max: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "NCPMetrics":
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class MPCMetrics:
    system_name: str = ""
    controller_type: str = ""  # "mppi" or "casadi"
    total_sim_time_seconds: float = 0.0
    solve_time_mean: float = 0.0
    solve_time_median: float = 0.0
    solve_time_max: float = 0.0
    # CasADi-specific
    ipopt_iterations_mean: float = 0.0
    convergence_failures: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MPCMetrics":
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class ClosedLoopMetrics:
    system_name: str = ""
    controller_type: str = ""  # "ncp_native", "ncp_step", "mppi_mpc", "casadi_mpc"
    num_ics: int = 0
    convergence_rate: float = 0.0  # fraction of ICs reaching target
    steps_to_converge_mean: float = 0.0
    steps_to_converge_median: float = 0.0
    trajectory_cost: float = 0.0  # mean ∫||x|| over converged trajectories
    final_error_mean: float = 0.0
    final_error_median: float = 0.0
    max_control_effort: float = 0.0
    control_energy: float = 0.0  # mean ∫||u||² over trajectories
    total_wall_time: float = 0.0
    solve_time_mean: float = 0.0
    solve_time_median: float = 0.0
    solve_time_max: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ClosedLoopMetrics":
        with open(path) as f:
            return cls(**json.load(f))


def compute_closed_loop_metrics(
    trajectories: torch.Tensor,   # (B, T, state_dim)
    controls: torch.Tensor,       # (B, T-1, ctrl_dim) or (B, T-1)
    converged: torch.Tensor,      # (B,) bool
    steps: torch.Tensor,          # (B,) int — steps taken
    solve_times: list[float],     # per-step solve times
    total_wall_time: float,
    system_name: str,
    controller_type: str,
) -> ClosedLoopMetrics:
    """Compute closed-loop metrics from simulation results."""
    B = trajectories.shape[0]
    num_converged = converged.sum().item()

    # Convergence rate
    convergence_rate = num_converged / B if B > 0 else 0.0

    # Steps to converge (only for converged trajectories)
    if num_converged > 0:
        conv_steps = steps[converged].float()
        steps_mean = conv_steps.mean().item()
        steps_median = conv_steps.median().item()
    else:
        steps_mean = steps_median = float("nan")

    # Trajectory cost: mean of sum(||x||) over time for each trajectory
    state_norms = trajectories.norm(dim=-1)  # (B, T)
    traj_costs = state_norms.sum(dim=-1)  # (B,)
    trajectory_cost = traj_costs.mean().item()

    # Final error
    final_states = torch.zeros(B, trajectories.shape[-1])
    for i in range(B):
        t = min(steps[i].item(), trajectories.shape[1] - 1)
        final_states[i] = trajectories[i, int(t)]
    final_errors = final_states.norm(dim=-1)
    final_error_mean = final_errors.mean().item()
    final_error_median = final_errors.median().item()

    # Control effort
    if controls.dim() == 2:
        ctrl_norms = controls.abs()  # (B, T-1) for 1D
    else:
        ctrl_norms = controls.norm(dim=-1)  # (B, T-1)
    max_control_effort = ctrl_norms.max().item()
    ctrl_energy = (ctrl_norms ** 2).sum(dim=-1).mean().item()

    # Solve times
    if solve_times:
        st = torch.tensor(solve_times)
        solve_time_mean = st.mean().item()
        solve_time_median = st.median().item()
        solve_time_max = st.max().item()
    else:
        solve_time_mean = solve_time_median = solve_time_max = 0.0

    return ClosedLoopMetrics(
        system_name=system_name,
        controller_type=controller_type,
        num_ics=B,
        convergence_rate=convergence_rate,
        steps_to_converge_mean=steps_mean,
        steps_to_converge_median=steps_median,
        trajectory_cost=trajectory_cost,
        final_error_mean=final_error_mean,
        final_error_median=final_error_median,
        max_control_effort=max_control_effort,
        control_energy=ctrl_energy,
        total_wall_time=total_wall_time,
        solve_time_mean=solve_time_mean,
        solve_time_median=solve_time_median,
        solve_time_max=solve_time_max,
    )
