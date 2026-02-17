"""Pendubot environment."""

from __future__ import annotations

from typing import Any

import torch

from ncp.dynamics.pendubot import pendubot_dynamics
from ncp.envs._base import BaseEnv, ControlSpec, StateSpec


class PendubotEnv(BaseEnv):
    """4-D pendubot environment.

    State: ``[q1, q2, q1_dot, q2_dot]``, control: torque on joint 1.
    Joint angles (dimensions 0 and 1) wrap to ``(-pi, pi]``.

    Args:
        num_envs: Number of parallel environments.
        dynamics_fn: Override dynamics (default: ``pendubot_dynamics``).
        max_torque: Symmetric torque bound on joint 1.
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
        max_torque: float = 5.0,
        dt: float = 0.05,
        alpha: float = 0.01,
        lipschitz: float = 5.0,
        eval_weights: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        dynamics_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dynamics_fn is None:
            dynamics_fn = pendubot_dynamics

        state_spec = StateSpec(dim=4, wrap_dims=(0, 1))
        control_spec = ControlSpec(
            dim=1,
            lower_bounds=torch.tensor([-max_torque]),
            upper_bounds=torch.tensor([max_torque]),
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
