"""Planar quadrotor environment."""

from __future__ import annotations

from typing import Any

import torch

from ncp.dynamics.quadrotor_2d import quadrotor_2d_dynamics
from ncp.envs._base import BaseEnv, ControlSpec, StateSpec


class Quadrotor2DEnv(BaseEnv):
    """6-D planar quadrotor (PVTOL) environment.

    State: ``[x, y, x_dot, y_dot, phi, phi_dot]``,
    control: ``[f1, f2]`` (left/right thrust).
    Roll angle phi (dimension 4) wraps to ``(-pi, pi]``.

    Args:
        num_envs: Number of parallel environments.
        dynamics_fn: Override dynamics (default: ``quadrotor_2d_dynamics``).
        mass: Quadrotor mass (used for thrust bounds).
        dt: Integration time step.
        alpha: Minimum contraction rate.
        lipschitz: Lipschitz constant.
        eval_weights: Optional per-dim weights for the evaluation metric.
        target: Target state.
        dynamics_kwargs: Extra kwargs forwarded to the dynamics function.
    """

    def __init__(
        self,
        num_envs: int = 10,
        *,
        dynamics_fn: Any = None,
        mass: float = 1.0,
        g: float = 9.81,
        dt: float = 0.02,
        alpha: float = 0.01,
        lipschitz: float = 5.0,
        eval_weights: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        dynamics_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dynamics_fn is None:
            dynamics_fn = quadrotor_2d_dynamics

        max_thrust = mass * g

        # Forward mass/g to dynamics so bounds and physics stay consistent
        if dynamics_kwargs is None:
            dynamics_kwargs = {}
        dynamics_kwargs.setdefault("m", mass)
        dynamics_kwargs.setdefault("g", g)

        state_spec = StateSpec(dim=6, wrap_dims=(4,))
        control_spec = ControlSpec(
            dim=2,
            lower_bounds=torch.tensor([0.0, 0.0]),
            upper_bounds=torch.tensor([max_thrust, max_thrust]),
        )

        super().__init__(
            num_envs=num_envs,
            state_spec=state_spec,
            control_spec=control_spec,
            dynamics_fn=dynamics_fn,
            dt=dt,
            alpha=alpha,
            lipschitz=lipschitz,
            eval_weights=eval_weights,
            target=target,
            dynamics_kwargs=dynamics_kwargs,
        )
