"""Base environment for NCP algorithm with configurable dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ncp.geometry.intersections import (
    count_hypercube_intersections,
    find_hypercube_intersections,
)
from ncp.utils.wrapping import wrap_angles


@dataclass(frozen=True)
class StateSpec:
    """Specification of the state space.

    Attributes:
        dim: State-space dimension.
        wrap_dims: Indices of angular dimensions that wrap to ``(-pi, pi]``.
            Pass an empty tuple for systems with no periodic dimensions.
    """

    dim: int
    wrap_dims: tuple[int, ...] = ()


@dataclass(frozen=True)
class ControlSpec:
    """Specification of the control space.

    Attributes:
        dim: Control dimension.
        lower_bounds: Per-dimension lower bounds (1-D tensor of length ``dim``).
        upper_bounds: Per-dimension upper bounds (1-D tensor of length ``dim``).
    """

    dim: int
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor


class BaseEnv:
    """System-agnostic environment used by the NCP builder.

    Subclasses only need to supply ``state_spec``, ``control_spec``, and
    a ``dynamics_fn`` (plus optional keyword arguments forwarded to the
    dynamics).  All trajectory sampling, MPPI search, and distance
    computation logic lives here.

    Args:
        num_envs: Number of parallel environment instances.
        state_spec: State-space specification.
        control_spec: Control-space specification.
        dynamics_fn: Callable ``(x, u, **kwargs) -> dx/dt`` where
            ``x`` is ``(B, state_dim)`` and ``u`` is ``(B, ctrl_dim)``.
        dt: Integration time step.
        alpha: Minimum contraction rate for the NCP certificate.
        lipschitz: Lipschitz constant *L*.
        eval_weights: Optional per-dimension weight tensor for the
            evaluation metric.  When ``None``, the inf-norm is used.
        target: Target state (defaults to origin).
        dynamics_kwargs: Extra keyword arguments forwarded to ``dynamics_fn``.
    """

    def __init__(
        self,
        num_envs: int,
        state_spec: StateSpec,
        control_spec: ControlSpec,
        dynamics_fn: Callable[..., torch.Tensor],
        *,
        dt: float = 0.1,
        alpha: float = 0.01,
        lipschitz: float = 5.0,
        eval_weights: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        dynamics_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = num_envs
        self.state_spec = state_spec
        self.control_spec = control_spec
        self.dynamics_fn = dynamics_fn
        self.dt = dt
        self.alpha = alpha
        self.lipschitz = lipschitz
        self.eval_weights = eval_weights
        self.dynamics_kwargs: dict[str, Any] = dynamics_kwargs or {}

        if target is None:
            self.target = torch.zeros(state_spec.dim, device=self.device)
        else:
            self.target = target.to(device=self.device, dtype=torch.float32)

        # Cache control bounds on device
        self._lb = self.control_spec.lower_bounds.to(self.device)
        self._ub = self.control_spec.upper_bounds.to(self.device)

        # Exponential cache (populated on first use by _get_exp_cache)
        self._exp_cache_steps: int = -1
        self._exp_alpha: torch.Tensor | None = None
        self._exp_combined: torch.Tensor | None = None

        self.states = self._sample_hypercube(num_envs)

    def _get_exp_cache(
        self, time_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached exponential tensors, recomputing only when T changes."""
        if self._exp_cache_steps != time_steps:
            t_eval = torch.linspace(
                0.0, time_steps * self.dt, steps=time_steps, device=self.device
            )
            self._exp_alpha = torch.exp(self.alpha * t_eval)
            self._exp_combined = torch.exp(
                (self.lipschitz + self.alpha) * t_eval
            )
            self._exp_cache_steps = time_steps
        return self._exp_alpha, self._exp_combined

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_hypercube(self, n: int, half_extent: float = 1.0) -> torch.Tensor:
        """Sample *n* points uniformly in ``[-half_extent, half_extent]^d``."""
        return (
            (torch.rand((n, self.state_spec.dim), device=self.device) * 2 - 1)
            * half_extent
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: torch.Tensor | None = None) -> None:
        """Reset environment states."""
        if seed is not None:
            self.states = seed
        else:
            self.states = self._sample_hypercube(self.num_envs)

    # ------------------------------------------------------------------
    # Dynamics integration
    # ------------------------------------------------------------------

    def _apply_dynamics(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """One Euler step: ``x + f(x, u) * dt``."""
        return x + self.dynamics_fn(x, u, **self.dynamics_kwargs) * self.dt

    def _clamp_controls(self, u: torch.Tensor) -> torch.Tensor:
        """Clamp each control dimension to its bounds."""
        return torch.max(torch.min(u, self._ub), self._lb)

    def _wrap(self, x: torch.Tensor, *, inplace: bool = False) -> torch.Tensor:
        """Wrap angular dimensions if any."""
        if self.state_spec.wrap_dims:
            return wrap_angles(x, self.state_spec.wrap_dims, inplace=inplace)
        return x

    # ------------------------------------------------------------------
    # Distance metric
    # ------------------------------------------------------------------

    def _distance(self, traj: torch.Tensor) -> torch.Tensor:
        """Compute distance from target along trailing state axis.

        Args:
            traj: ``(..., D)`` tensor.

        Returns:
            ``(...)`` tensor of scalar distances.
        """
        diff = traj - self.target
        if self.eval_weights is not None:
            w = self.eval_weights.to(device=diff.device)
            return torch.sqrt((diff**2 * w).sum(dim=-1))
        # abs().max() is faster than linalg.norm(..., ord=inf) for small D
        return diff.abs().max(dim=-1).values

    # ------------------------------------------------------------------
    # Trajectory rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def trajectories(
        self,
        controls: torch.Tensor,
        sentinel: float = 1337.0,
    ) -> torch.Tensor:
        """Roll out trajectories from current states.

        Args:
            controls: ``(num_envs, T, ctrl_dim)`` or ``(num_envs, T)``
                for 1-D control systems.  Sentinel values mark no-op steps.
            sentinel: Value used to mark inactive time steps.

        Returns:
            ``(num_envs, T+1, state_dim)`` trajectory tensor.
        """
        if controls.dim() == 2 and self.control_spec.dim == 1:
            controls = controls.unsqueeze(-1)

        B = controls.shape[0]
        time_steps = controls.shape[1]
        state_dim = self.state_spec.dim

        # Pre-allocate trajectory tensor
        traj = torch.empty(B, time_steps + 1, state_dim, device=self.device)
        current = self.states.clone()
        traj[:, 0] = current

        for t in range(time_steps):
            u_t = controls[:, t]
            if self.control_spec.dim == 1:
                mask = u_t.squeeze(-1) != sentinel
            else:
                mask = (u_t != sentinel).any(dim=-1)

            if mask.all():
                current = self._apply_dynamics(current, u_t)
            elif mask.any():
                next_states = current.clone()
                next_states[mask] = self._apply_dynamics(current[mask], u_t[mask])
                current = next_states
            traj[:, t + 1] = current

        return traj

    # ------------------------------------------------------------------
    # MPPI-style trajectory sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_trajectory(
        self,
        time_steps: int = 5,
        control_seed: torch.Tensor | None = None,
        variance: float = 1.0,
        num_samples: int = 1,
        r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample trajectories via MPPI and return the best per environment.

        Returns:
            ``(best_controls, best_trajectories, best_distances, best_taus)``
        """
        ctrl_dim = self.control_spec.dim
        state_dim = self.state_spec.dim
        device = self.device
        num_envs = self.num_envs
        total = num_envs * num_samples

        if r is None:
            r = torch.zeros(num_envs, device=device)

        batch_states = self.states.clone().repeat_interleave(num_samples, dim=0)

        # Build control samples
        if ctrl_dim == 1:
            ctrl_shape = (num_envs, num_samples, time_steps)
        else:
            ctrl_shape = (num_envs, num_samples, time_steps, ctrl_dim)

        if control_seed is not None:
            mu = control_seed.unsqueeze(1).expand(*ctrl_shape)
        else:
            mu = torch.zeros(ctrl_shape, device=device)

        action_lists = torch.normal(mu, variance)

        if ctrl_dim == 1:
            action_lists = action_lists.view(-1, time_steps)
            action_lists.clamp_(self._lb.item(), self._ub.item())
        else:
            action_lists = action_lists.view(-1, time_steps, ctrl_dim)
            action_lists = torch.max(torch.min(action_lists, self._ub), self._lb)

        # Pre-allocated trajectory rollout
        traj = torch.empty(total, time_steps + 1, state_dim, device=device)
        traj[:, 0] = batch_states

        for t in range(time_steps):
            batch_states = batch_states + self.dynamics_fn(
                batch_states, action_lists[:, t], **self.dynamics_kwargs
            ) * self.dt
            traj[:, t + 1] = batch_states

        traj = self._wrap(traj, inplace=True)
        dist = self._distance(traj[:, 1:])

        # Alpha scoring with cached exponentials — combined min+argmin
        r_rep = r.repeat_interleave(num_samples)
        exp_alpha, exp_combined = self._get_exp_cache(time_steps)
        alpha_dist = dist * exp_alpha + r_rep.unsqueeze(1) * exp_combined

        # Single pass: min returns both values and indices
        min_result = alpha_dist.min(dim=1)
        alpha_distances = min_result.values.view(num_envs, num_samples)
        taus = min_result.indices.view(num_envs, num_samples)

        best_indices = alpha_distances.argmin(dim=1)
        env_range = torch.arange(num_envs, device=device)
        best_taus = taus[env_range, best_indices]
        best_dist = alpha_distances[env_range, best_indices]

        if ctrl_dim == 1:
            best_actions = action_lists.view(
                num_envs, num_samples, time_steps
            )[env_range, best_indices]
        else:
            best_actions = action_lists.view(
                num_envs, num_samples, time_steps, ctrl_dim
            )[env_range, best_indices]

        best_traj = traj.view(
            num_envs, num_samples, time_steps + 1, state_dim
        )[env_range, best_indices]

        return best_actions, best_traj, best_dist, best_taus

    # ------------------------------------------------------------------
    # MPPI with reuse (bootstrapping from verified cells)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_trajectory_reuse_new(
        self,
        time_steps: int = 5,
        control_seed: torch.Tensor | None = None,
        variance: float = 1.0,
        num_samples: int = 1,
        r: torch.Tensor | None = None,
        centers: torch.Tensor | None = None,
        radii: torch.Tensor | None = None,
        verified_alphas: torch.Tensor | None = None,
        verified_controls: torch.Tensor | None = None,
        verified_indices: torch.Tensor | None = None,
        splits: torch.Tensor | None = None,
        unverified_centers: torch.Tensor | None = None,
        unverified_radii: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MPPI sampling with bootstrapping from previously verified cells.

        Returns:
            ``(best_controls, best_traj, best_dist, best_taus, class_success)``
        """
        ctrl_dim = self.control_spec.dim
        state_dim = self.state_spec.dim
        device = self.device
        lipschitz = self.lipschitz
        total = self.num_envs * num_samples

        if r is None:
            r = torch.zeros(self.num_envs, device=device)
        r = r.repeat_interleave(num_samples)

        batch_states = self.states.clone().repeat_interleave(num_samples, dim=0)

        class_ids = torch.arange(total, device=device) // num_samples
        class_success = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        active_mask = torch.ones(total, dtype=torch.bool, device=device)
        taus = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        best_indices = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        # Build control samples
        if ctrl_dim == 1:
            ctrl_shape = (self.num_envs, num_samples, time_steps)
        else:
            ctrl_shape = (self.num_envs, num_samples, time_steps, ctrl_dim)

        if control_seed is not None:
            mu = control_seed.unsqueeze(1).expand(*ctrl_shape)
        else:
            mu = torch.zeros(ctrl_shape, device=device)

        action_lists = torch.normal(mu, variance)

        if ctrl_dim == 1:
            action_lists = action_lists.view(-1, time_steps)
            action_lists.clamp_(self._lb.item(), self._ub.item())
        else:
            action_lists = action_lists.view(-1, time_steps, ctrl_dim)
            action_lists = torch.max(torch.min(action_lists, self._ub), self._lb)

        can_bootstrap = (
            centers is not None
            and radii is not None
            and verified_controls is not None
            and verified_indices is not None
        )

        # Pre-compute bootstrap threshold
        if can_bootstrap:
            r_min = r.min().item()
            radii_max = radii.max().item()

        # Pre-allocate trajectory tensor
        traj = torch.empty(total, time_steps + 1, state_dim, device=device)
        traj[:, 0] = batch_states

        for t in range(time_steps):
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            u_t = action_lists[active_indices, t]
            movement = self.dynamics_fn(
                batch_states[active_indices], u_t, **self.dynamics_kwargs
            )
            batch_states = batch_states.clone()
            batch_states[active_indices] += movement * self.dt
            traj[:, t + 1] = batch_states

            if (
                can_bootstrap
                and r_min * math.exp(lipschitz * t * self.dt) < 2 * radii_max
            ):
                idx = _check_parallel(
                    batch_states[active_indices],
                    t,
                    r[active_indices],
                    self.alpha,
                    lipschitz,
                    self.dt,
                    num_samples,
                    unverified_centers,
                    unverified_radii,
                    centers,
                    radii,
                    verified_alphas,
                    verified_indices,
                    traj[:, :t + 2],
                    self.state_spec.wrap_dims,
                )

                global_success_indices = active_indices[idx]
                global_success_classes = class_ids[global_success_indices]

                new_success_class_mask = ~class_success[global_success_classes]
                new_success_indices = global_success_indices[new_success_class_mask]
                new_success_classes = global_success_classes[new_success_class_mask]

                class_success[new_success_classes] = True
                best_indices[new_success_classes] = new_success_indices
                taus[new_success_classes] = t

                active_mask = active_mask & (~class_success[class_ids])

        mask = ~class_success

        traj = self._wrap(traj, inplace=True)

        dist = self._distance(traj[:, 1:])
        exp_alpha, exp_combined = self._get_exp_cache(time_steps)
        alpha_dist = dist * exp_alpha + r.unsqueeze(1) * exp_combined

        min_result = alpha_dist.min(dim=1)
        alpha_distances = min_result.values
        new_taus = min_result.indices

        alpha_distances = alpha_distances.view(self.num_envs, num_samples)
        new_taus = new_taus.view(self.num_envs, num_samples)

        env_range = torch.arange(self.num_envs, device=device)
        new_best = alpha_distances.argmin(dim=1)
        best_indices[mask] = new_best[mask]
        best_indices = best_indices % num_samples
        taus[mask] = new_taus[env_range, best_indices][mask]

        if ctrl_dim == 1:
            best_actions = action_lists.view(
                self.num_envs, num_samples, time_steps
            )[env_range, best_indices]
        else:
            best_actions = action_lists.view(
                self.num_envs, num_samples, time_steps, ctrl_dim
            )[env_range, best_indices]

        best_traj = traj.view(
            self.num_envs, num_samples, time_steps + 1, state_dim
        )[env_range, best_indices]

        return (
            best_actions,
            best_traj,
            alpha_distances.min(dim=1).values,
            taus,
            class_success,
        )


# ------------------------------------------------------------------
# check_parallel (module-level helper)
# ------------------------------------------------------------------


def _check_parallel(
    batch_states: torch.Tensor,
    t: int,
    r: torch.Tensor,
    alpha: float,
    lipschitz: float,
    rate: float,
    num_samples: int,
    unverified_centers: torch.Tensor | None,
    unverified_radii: torch.Tensor | None,
    verified_centers: torch.Tensor,
    verified_radii: torch.Tensor,
    verified_alphas: torch.Tensor,
    verified_indices: torch.Tensor,
    trajectories: torch.Tensor,
    wrap_dims: tuple[int, ...] = (),
) -> torch.Tensor:
    """Check which sample trajectories can bootstrap into verified cells."""
    device = batch_states.device
    B = batch_states.shape[0]

    scale_factor = math.exp(lipschitz * t * rate)
    radii_tensor = r * scale_factor

    if unverified_centers is not None and unverified_radii is not None:
        unverified_count = count_hypercube_intersections(
            unverified_centers,
            unverified_radii,
            batch_states,
            radii_tensor,
            wrap_dims=wrap_dims,
        )
        eligible_mask = unverified_count == 0
    else:
        eligible_mask = torch.ones(B, dtype=torch.bool, device=device)

    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).squeeze(1)
    if eligible_indices.numel() == 0:
        return eligible_indices.new_empty(0)

    query_states = batch_states[eligible_indices]
    query_radii = radii_tensor[eligible_indices]

    verified_lists = find_hypercube_intersections(
        verified_centers,
        verified_radii,
        query_states,
        query_radii,
        wrap_dims=wrap_dims,
    )

    k_offsets = torch.arange(len(verified_lists), device=device)
    row_counts = torch.tensor(
        [v.numel() for v in verified_lists], dtype=torch.long, device=device
    )
    if row_counts.sum() == 0:
        return eligible_indices.new_empty(0)

    k_idx = torch.repeat_interleave(k_offsets, row_counts)
    k_i_idx = torch.cat(verified_lists, dim=0)

    v_centers = verified_centers[k_i_idx]
    v_radii = verified_radii[k_i_idx]
    v_alphas = verified_alphas[k_i_idx]
    v_indices = verified_indices[k_i_idx]

    # Initial state of each matched trajectory (all state dims at t=0)
    # Use inf-norm consistent with _distance / the main algorithm's V(x)
    traj_k = trajectories[eligible_indices[k_idx], 0, :]

    norm_traj = traj_k.abs().max(dim=1).values - r[eligible_indices][k_idx]
    term1 = torch.exp(-(v_alphas - alpha) * v_indices * rate)
    term2 = torch.exp(v_alphas * t * rate)
    numerator = v_centers.abs().max(dim=1).values + v_radii
    inequality = term1 * term2 * numerator / norm_traj

    failed = inequality > 1
    failed_mask = torch.zeros(
        len(verified_lists), dtype=torch.bool, device=device
    )
    failed_mask.index_put_(
        (k_idx[failed],),
        torch.ones_like(k_idx[failed], dtype=torch.bool),
        accumulate=True,
    )

    passed_mask = ~failed_mask
    return eligible_indices[passed_mask]
