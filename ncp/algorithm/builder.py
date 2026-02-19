"""NCPBuilder — system-agnostic NCP algorithm driver."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Type

import torch

from ncp.algorithm.mppi import find_path, find_path_reuse
from ncp.algorithm.result import NCPResult
from ncp.algorithm.search import search_alpha_parallel
from ncp.envs._base import BaseEnv, ControlSpec, StateSpec
from ncp.geometry.grid import generate_initial_grid, subdivide_cells
from ncp.utils.wrapping import wrap_angles


class _GrowableBuffer:
    """Amortised-doubling tensor buffer to avoid O(n^2) torch.cat growth."""

    __slots__ = ("_buf", "_size")

    def __init__(self, initial: torch.Tensor) -> None:
        self._buf = initial.clone()
        self._size = initial.shape[0]

    def append(self, t: torch.Tensor) -> None:
        n = t.shape[0]
        if n == 0:
            return
        needed = self._size + n
        if needed > self._buf.shape[0]:
            new_cap = max(needed, self._buf.shape[0] * 2)
            new_buf = torch.empty(
                (new_cap, *self._buf.shape[1:]),
                dtype=self._buf.dtype,
                device=self._buf.device,
            )
            new_buf[: self._size] = self._buf[: self._size]
            self._buf = new_buf
        self._buf[self._size : self._size + n] = t
        self._size += n

    def tensor(self) -> torch.Tensor:
        return self._buf[: self._size]

    def __len__(self) -> int:
        return self._size


class NCPBuilder:
    """Build an NCP (Nonparametric Chain Policy) for a given dynamical system.

    The builder is system-agnostic: it accepts an environment factory and
    dynamics function and handles the adaptive grid refinement loop.

    Args:
        env_factory: A :class:`BaseEnv` subclass (or callable returning one).
        state_spec: State-space specification.
        control_spec: Control-space specification.
        dynamics_fn: Dynamics function ``(x, u, **kw) -> dx/dt``.
        radius: Outer radius of the region to cover.
        epsilon: Finest resolution half-width.
        lipschitz: Lipschitz constant *L*.
        tau: Trajectory horizon time.
        dt: Integration time step.
        min_alpha: Minimum acceptable contraction rate.
        max_splits: Maximum subdivision depth.
        batch_size: Cells processed per batch.
        num_samples: MPPI samples per iteration.
        reuse: Enable bootstrapping from verified cells.
        dynamics_kwargs: Extra kwargs forwarded to the dynamics.
        angle_bound_dims: Dimensions whose centres must satisfy
            ``|center_i| - radius <= pi`` (for angular state spaces).
    """

    def __init__(
        self,
        env_factory: Type[BaseEnv] | Callable[..., BaseEnv],
        state_spec: StateSpec,
        control_spec: ControlSpec,
        dynamics_fn: Callable[..., torch.Tensor],
        *,
        radius: float = 2.0,
        epsilon: float = 0.01,
        lipschitz: float = 5.0,
        tau: float = 2.0,
        dt: float = 0.1,
        min_alpha: float = 0.01,
        max_splits: int = 3,
        batch_size: int = 500,
        num_samples: int = 1000,
        reuse: bool = False,
        dynamics_kwargs: dict[str, Any] | None = None,
        angle_bound_dims: tuple[int, ...] | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.state_spec = state_spec
        self.control_spec = control_spec
        self.dynamics_fn = dynamics_fn
        self.radius = radius
        self.epsilon = epsilon
        self.lipschitz = lipschitz
        self.tau = tau
        self.dt = dt
        self.min_alpha = min_alpha
        self.max_splits = max_splits
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.reuse = reuse
        self.dynamics_kwargs = dynamics_kwargs or {}
        # By default, apply angle bounds on wrap_dims
        self.angle_bound_dims = (
            angle_bound_dims
            if angle_bound_dims is not None
            else state_spec.wrap_dims
        )

    @torch.no_grad()
    def build(
        self,
        checkpoint_fn: Callable[[NCPResult], None] | None = None,
        checkpoint_interval: float = 300.0,
        **extra_dynamics_kwargs: Any,
    ) -> NCPResult:
        """Run the NCP algorithm and return the result.

        Args:
            checkpoint_fn: If provided, called periodically with the current
                partial :class:`NCPResult`. Useful for saving intermediate
                results so that progress survives timeouts.
            checkpoint_interval: Seconds between checkpoint calls (default 300).
            **extra_dynamics_kwargs: Merged into ``dynamics_kwargs``.

        Returns:
            :class:`NCPResult` containing all verified (and optionally
            unverified) cells.
        """
        dyn_kw = {**self.dynamics_kwargs, **extra_dynamics_kwargs}
        d = self.state_spec.dim
        ctrl_dim = self.control_spec.dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        time_steps = int(self.tau / self.dt)
        t_eval = torch.linspace(0.0, self.tau, steps=time_steps + 1, device=device)
        wrap_dims = self.state_spec.wrap_dims

        start = time.time()

        # ---- Initial grid ----
        points, radii = generate_initial_grid(
            d, self.epsilon, self.radius, device=device
        )

        splits = torch.zeros_like(radii)
        if ctrl_dim == 1:
            controls = torch.zeros(len(points), time_steps, device=device)
        else:
            controls = torch.zeros(
                len(points), time_steps, ctrl_dim, device=device
            )

        # Apply angular-dimension bounds to initial grid
        if self.angle_bound_dims:
            keep = torch.ones(len(points), dtype=torch.bool, device=device)
            for dim_idx in self.angle_bound_dims:
                keep &= torch.abs(points[:, dim_idx]) - radii <= math.pi
            points = points[keep]
            radii = radii[keep]
            controls = controls[keep]
            splits = splits[keep]

        # ---- Verified set (seed with origin) ----
        vc_init = torch.zeros(1, d, device=device)
        vr_init = torch.tensor([self.epsilon], device=device)
        if ctrl_dim == 1:
            vctrl_init = torch.zeros(1, time_steps, device=device)
        else:
            vctrl_init = torch.zeros(1, time_steps, ctrl_dim, device=device)
        va_init = torch.zeros(1, device=device)
        vi_init = torch.ones(1, device=device)

        buf_centers = _GrowableBuffer(vc_init)
        buf_radii = _GrowableBuffer(vr_init)
        buf_controls = _GrowableBuffer(vctrl_init)
        buf_alphas = _GrowableBuffer(va_init)
        buf_indices = _GrowableBuffer(vi_init)

        unverified_centers: torch.Tensor | None = None
        unverified_radii: torch.Tensor | None = None
        last_checkpoint = start

        # ---- Main loop ----
        while len(points) > 0:
            batch_pts = points[: self.batch_size]
            batch_rad = radii[: self.batch_size]
            batch_ctrl = controls[: self.batch_size]
            batch_splits = splits[: self.batch_size]

            points = points[self.batch_size :]
            radii = radii[self.batch_size :]
            controls = controls[self.batch_size :]
            splits = splits[self.batch_size :]

            env = self.env_factory(
                num_envs=len(batch_pts),
                dt=self.dt,
                alpha=self.min_alpha,
                lipschitz=self.lipschitz,
                dynamics_kwargs=dyn_kw,
            )
            env.reset(batch_pts)

            # Materialise current verified tensors for reuse path
            if self.reuse:
                verified_centers = buf_centers.tensor()
                verified_radii = buf_radii.tensor()
                verified_alphas = buf_alphas.tensor()
                verified_controls = buf_controls.tensor()
                verified_indices = buf_indices.tensor()

                winner, _, taus, reusing = find_path_reuse(
                    env,
                    seeds=batch_pts,
                    time_steps=time_steps,
                    control_seed=batch_ctrl,
                    num_samples=self.num_samples,
                    r=batch_rad,
                    centers=verified_centers,
                    radii=verified_radii,
                    verified_alphas=verified_alphas,
                    verified_controls=verified_controls,
                    verified_indices=verified_indices,
                    splits=batch_splits,
                    unverified_centers=(
                        torch.cat((points, batch_pts))
                        if len(points) > 0
                        else batch_pts
                    ),
                    unverified_radii=(
                        torch.cat((radii, batch_rad))
                        if len(radii) > 0
                        else batch_rad
                    ),
                )
            else:
                winner, _, taus = find_path(
                    env,
                    seeds=batch_pts,
                    time_steps=time_steps,
                    control_seed=batch_ctrl,
                    num_samples=self.num_samples,
                    r=batch_rad,
                )

            env.reset(batch_pts)
            sol = env.trajectories(winner)
            if wrap_dims:
                sol = wrap_angles(sol, wrap_dims)
            sol_norms = env._distance(sol)

            if self.reuse and reusing.any():
                alpha = self.min_alpha * torch.ones_like(batch_rad)
                indices = taus.clone()
                non_reusing = ~reusing
                if non_reusing.sum() > 0:
                    alpha_calc, idx_calc = search_alpha_parallel(
                        sol_norms[non_reusing],
                        env._distance(batch_pts)[non_reusing],
                        batch_rad[non_reusing],
                        t_eval,
                        self.lipschitz,
                    )
                    alpha[non_reusing] = alpha_calc
                    indices[non_reusing] = idx_calc
            else:
                alpha, indices = search_alpha_parallel(
                    sol_norms,
                    env._distance(batch_pts),
                    batch_rad,
                    t_eval,
                    self.lipschitz,
                )

            indices = indices + 1
            mask = alpha >= self.min_alpha

            buf_centers.append(batch_pts[mask])
            buf_radii.append(batch_rad[mask])
            buf_controls.append(winner[mask])
            buf_alphas.append(alpha[mask])
            buf_indices.append(indices[mask])

            # ---- Subdivide failed cells ----
            n_failed = (~mask).sum().item()
            if n_failed > 0:
                new_centers, new_radii = subdivide_cells(
                    batch_pts[~mask], batch_rad[~mask], d
                )
                sub_count = 3 ** d
                new_splits = batch_splits[~mask].repeat_interleave(sub_count) + 1
                new_controls = winner[~mask].repeat_interleave(sub_count, dim=0)

                points = torch.cat((points, new_centers), 0)
                radii = torch.cat((radii, new_radii), 0)
                controls = torch.cat((controls, new_controls), 0)
                splits = torch.cat((splits, new_splits), 0)

                # Filter by bounds
                keep = torch.ones(len(points), dtype=torch.bool, device=device)

                # Inf-norm bound
                keep &= (
                    points.abs().max(dim=1).values - radii <= self.radius
                )

                # Angular-dimension bounds
                for dim_idx in self.angle_bound_dims:
                    keep &= torch.abs(points[:, dim_idx]) - radii <= math.pi

                # Max splits
                keep &= splits < self.max_splits

                # Track unverified (exceeded max splits)
                exceeded = ~keep & (splits >= self.max_splits)
                if exceeded.any():
                    if unverified_centers is None:
                        unverified_centers = points[exceeded]
                        unverified_radii = radii[exceeded]
                    else:
                        unverified_centers = torch.cat(
                            (unverified_centers, points[exceeded]), dim=0
                        )
                        unverified_radii = torch.cat(
                            (unverified_radii, radii[exceeded]), dim=0
                        )

                points = points[keep]
                radii = radii[keep]
                controls = controls[keep]
                splits = splits[keep]

            # ---- Periodic checkpoint ----
            now = time.time()
            if checkpoint_fn is not None and now - last_checkpoint >= checkpoint_interval:
                last_checkpoint = now
                checkpoint_fn(NCPResult(
                    verified_centers=buf_centers.tensor(),
                    verified_radii=buf_radii.tensor(),
                    verified_controls=buf_controls.tensor(),
                    verified_alphas=buf_alphas.tensor(),
                    verified_indices=buf_indices.tensor(),
                    unverified_centers=unverified_centers,
                    unverified_radii=unverified_radii,
                    elapsed_seconds=now - start,
                ))

        elapsed = time.time() - start

        return NCPResult(
            verified_centers=buf_centers.tensor(),
            verified_radii=buf_radii.tensor(),
            verified_controls=buf_controls.tensor(),
            verified_alphas=buf_alphas.tensor(),
            verified_indices=buf_indices.tensor(),
            unverified_centers=unverified_centers,
            unverified_radii=unverified_radii,
            elapsed_seconds=elapsed,
        )
