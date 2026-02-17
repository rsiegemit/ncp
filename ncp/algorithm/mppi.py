"""MPPI-based path finding for NCP."""

from __future__ import annotations

import torch

from ncp.envs._base import BaseEnv


@torch.no_grad()
def find_path(
    env: BaseEnv,
    *,
    seeds: torch.Tensor | None = None,
    countermax: int = 1,
    num_samples: int = 1000,
    variance: float = 1.0,
    time_steps: int = 20,
    control_seed: torch.Tensor | None = None,
    r: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run MPPI iterations to find control sequences from given seeds.

    Args:
        env: Environment instance (must have ``sample_trajectory``).
        seeds: ``(num_envs, D)`` starting states.
        countermax: Iterations without improvement before stopping.
        num_samples: Controls sampled per iteration.
        variance: Sampling noise.
        time_steps: Trajectory horizon.
        control_seed: Initial control guess.
        r: ``(num_envs,)`` per-cell radii for alpha evaluation.

    Returns:
        ``(best_controls, iterations, best_taus)``
    """
    device = env.device
    num_envs = env.num_envs

    if seeds is None:
        seeds = env._sample_hypercube(num_envs)

    env.states = seeds.clone()

    winner = control_seed
    endval = torch.full((num_envs,), float("inf"), device=device)
    counter = torch.zeros(num_envs, device=device)
    iters = torch.zeros(num_envs, dtype=torch.int32, device=device)
    real_taus = torch.zeros(num_envs, dtype=torch.int32, device=device)

    while torch.any(counter < countermax):
        env.states = seeds.clone()
        sol_actions, _, sol_distances, taus = env.sample_trajectory(
            time_steps=time_steps,
            control_seed=winner,
            num_samples=num_samples,
            variance=variance,
            r=r,
        )

        improved = sol_distances < endval
        endval = torch.where(improved, sol_distances, endval)
        if winner is None:
            winner = sol_actions
        else:
            if winner.dim() == sol_actions.dim():
                winner = torch.where(
                    improved.view(-1, *([1] * (sol_actions.dim() - 1))),
                    sol_actions,
                    winner,
                )
            else:
                winner = sol_actions
        real_taus = torch.where(improved, taus, real_taus)

        counter[improved] = 0
        counter[~improved] += 1
        iters += 1

    env.states = seeds.clone()
    return winner, iters, real_taus


@torch.no_grad()
def find_path_reuse(
    env: BaseEnv,
    *,
    seeds: torch.Tensor | None = None,
    countermax: int = 1,
    num_samples: int = 1000,
    variance: float = 1.0,
    time_steps: int = 20,
    control_seed: torch.Tensor | None = None,
    r: torch.Tensor | None = None,
    centers: torch.Tensor | None = None,
    radii: torch.Tensor | None = None,
    verified_alphas: torch.Tensor | None = None,
    verified_controls: torch.Tensor | None = None,
    verified_indices: torch.Tensor | None = None,
    splits: torch.Tensor | None = None,
    unverified_centers: torch.Tensor | None = None,
    unverified_radii: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MPPI path-finding with bootstrapping from verified cells.

    Returns:
        ``(best_controls, iterations, best_taus, class_success)``
    """
    device = env.device
    num_envs = env.num_envs

    if seeds is None:
        seeds = env._sample_hypercube(num_envs)

    env.states = seeds.clone()

    winner = control_seed
    endval = torch.full((num_envs,), float("inf"), device=device)
    counter = torch.zeros(num_envs, device=device)
    iters = torch.zeros(num_envs, dtype=torch.int32, device=device)
    real_taus = torch.zeros(num_envs, dtype=torch.int32, device=device)
    reusing = torch.zeros(num_envs, dtype=torch.bool, device=device)

    while torch.any(counter < countermax):
        env.states = seeds.clone()
        if centers is not None and splits is not None:
            sol_actions, _, sol_distances, taus, reusing = (
                env.sample_trajectory_reuse_new(
                    time_steps=time_steps,
                    control_seed=winner,
                    num_samples=num_samples,
                    variance=variance,
                    r=r,
                    centers=centers,
                    radii=radii,
                    verified_alphas=verified_alphas,
                    verified_controls=verified_controls,
                    verified_indices=verified_indices,
                    splits=splits,
                    unverified_centers=unverified_centers,
                    unverified_radii=unverified_radii,
                )
            )
        else:
            sol_actions, _, sol_distances, taus = env.sample_trajectory(
                time_steps=time_steps,
                control_seed=winner,
                num_samples=num_samples,
                variance=variance,
                r=r,
            )

        improved = sol_distances < endval
        endval = torch.where(improved, sol_distances, endval)
        if winner is None:
            winner = sol_actions
        else:
            if winner.dim() == sol_actions.dim():
                winner = torch.where(
                    improved.view(-1, *([1] * (sol_actions.dim() - 1))),
                    sol_actions,
                    winner,
                )
            else:
                winner = sol_actions
        real_taus = torch.where(improved, taus, real_taus)

        counter[improved] = 0
        counter[~improved] += 1
        iters += 1

    env.states = seeds.clone()
    return winner, iters, real_taus, reusing
