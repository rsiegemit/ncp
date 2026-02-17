"""Duffing oscillator environment."""

from __future__ import annotations

from typing import Any

import torch

from ncp.dynamics.duffing import duffing_dynamics
from ncp.envs._base import BaseEnv, ControlSpec, StateSpec


class DuffingEnv(BaseEnv):
    """2-D Duffing oscillator environment.

    State: ``[x, x_dot]``, control: external force.
    No angular wrapping.

    Args:
        num_envs: Number of parallel environments.
        dynamics_fn: Override dynamics (default: ``duffing_dynamics``).
        max_force: Symmetric force bound.
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
        max_force: float = 10.0,
        dt: float = 0.05,
        alpha: float = 0.01,
        lipschitz: float = 5.0,
        eval_weights: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        dynamics_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dynamics_fn is None:
            dynamics_fn = duffing_dynamics

        state_spec = StateSpec(dim=2, wrap_dims=())
        control_spec = ControlSpec(
            dim=1,
            lower_bounds=torch.tensor([-max_force]),
            upper_bounds=torch.tensor([max_force]),
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
