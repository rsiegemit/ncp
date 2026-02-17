"""Unicycle environment."""

from __future__ import annotations

from typing import Any

import torch

from ncp.dynamics.unicycle import unicycle_derivatives
from ncp.envs._base import BaseEnv, ControlSpec, StateSpec


class UnicycleEnv(BaseEnv):
    """3-D unicycle environment.

    State: ``[x, y, theta]``, control: ``[v, omega]``.
    Theta (dimension 2) wraps to ``(-pi, pi]``.
    Forward velocity *v* is clamped to ``[0, max_speed]``.

    Args:
        num_envs: Number of parallel environments.
        dynamics_fn: Override dynamics (default: ``unicycle_derivatives``).
        max_speed: Maximum linear/angular speed.
        dt: Integration time step.
        alpha: Minimum contraction rate.
        lipschitz: Lipschitz constant.
        eval_weights: Optional per-dim weights for the evaluation metric.
        target: Target state.
        dynamics_kwargs: Extra kwargs forwarded to the dynamics function
            (e.g. ``{"wind": True, "windval": -0.6}``).
    """

    def __init__(
        self,
        num_envs: int = 10,
        *,
        dynamics_fn: Any = None,
        max_speed: float = 1.0,
        dt: float = 0.1,
        alpha: float = 0.01,
        lipschitz: float = 5.0,
        eval_weights: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        dynamics_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dynamics_fn is None:
            dynamics_fn = unicycle_derivatives

        state_spec = StateSpec(dim=3, wrap_dims=(2,))
        control_spec = ControlSpec(
            dim=2,
            lower_bounds=torch.tensor([0.0, -max_speed]),
            upper_bounds=torch.tensor([max_speed, max_speed]),
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
