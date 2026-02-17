"""Simulation / evaluation loop for NCP controllers."""

from __future__ import annotations

import torch

from ncp.envs._base import BaseEnv
from ncp.evaluation.controller import NCPController
from ncp.utils.wrapping import wrap_angles


@torch.no_grad()
def simulate(
    env: BaseEnv,
    controller: NCPController,
    initial_states: torch.Tensor,
    *,
    max_steps: int = 1000,
    precision: float = 0.03,
) -> dict[str, torch.Tensor]:
    """Simulate the NCP controller from initial states until convergence.

    At each step the controller looks up the current state in the
    verified grid and executes the stored control. The simulation
    terminates for each trajectory once the state is within *precision*
    of the target (inf-norm) or *max_steps* is reached.

    Args:
        env: An environment instance (used for dynamics and wrapping).
        controller: An :class:`NCPController` built from NCP results.
        initial_states: ``(B, D)`` starting states.
        max_steps: Safety limit on total integration steps.
        precision: Convergence tolerance.

    Returns:
        A dict with keys:
        - ``"trajectories"``: ``(B, T_actual, D)`` state trajectories.
        - ``"converged"``: ``(B,)`` boolean mask of converged runs.
        - ``"steps"``: ``(B,)`` number of steps taken per trajectory.
    """
    device = initial_states.device
    B, D = initial_states.shape
    wrap_dims = env.state_spec.wrap_dims

    points = initial_states.clone()
    converged = torch.zeros(B, dtype=torch.bool, device=device)
    steps = torch.zeros(B, dtype=torch.long, device=device)
    all_traj = [points.clone()]

    # Reuse a single env for trajectory rollout instead of creating new ones
    sim_env = env.__class__(
        num_envs=B,
        dt=env.dt,
        alpha=env.alpha,
        lipschitz=env.lipschitz,
        dynamics_kwargs=env.dynamics_kwargs,
    )

    for _ in range(max_steps):
        if converged.all():
            break

        active = ~converged
        active_pts = points[active]

        if wrap_dims:
            active_pts = wrap_angles(active_pts, wrap_dims)

        ctrl, taus = controller.lookup(active_pts)

        n_active = active_pts.shape[0]
        time_steps = ctrl.shape[1]

        # Resize sim_env if needed, otherwise just reset
        if sim_env.num_envs != n_active:
            sim_env = env.__class__(
                num_envs=n_active,
                dt=env.dt,
                alpha=env.alpha,
                lipschitz=env.lipschitz,
                dynamics_kwargs=env.dynamics_kwargs,
            )
        sim_env.reset(active_pts)
        traj = sim_env.trajectories(ctrl, sentinel=controller.sentinel)

        # Vectorised: gather final position at tau for each trajectory
        # taus: (n_active,) — clamp to valid range
        t_idx = taus.clamp(0, time_steps).long()
        new_pts = traj[torch.arange(n_active, device=device), t_idx]

        if wrap_dims:
            new_pts = wrap_angles(new_pts, wrap_dims)

        points = points.clone()
        points[active] = new_pts
        all_traj.append(points.clone())

        dist = (points - env.target).abs().max(dim=1).values
        just_converged = (~converged) & (dist < precision)
        converged = converged | just_converged
        steps[~converged] += 1

    return {
        "trajectories": torch.stack(all_traj, dim=1),
        "converged": converged,
        "steps": steps,
    }
