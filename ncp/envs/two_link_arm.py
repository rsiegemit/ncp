"""2-link robot arm environment."""

from __future__ import annotations

from typing import Any

import torch

from ncp.dynamics.two_link_arm import two_link_arm_dynamics
from ncp.envs._base import BaseEnv, ControlSpec, StateSpec


class TwoLinkArmEnv(BaseEnv):
    """4-D fully-actuated 2-link robot arm environment.

    State: ``[q1, q2, q1_dot, q2_dot]``, control: ``[tau1, tau2]``.
    Joint angles (dimensions 0 and 1) wrap to ``(-pi, pi]``.

    Args:
        num_envs: Number of parallel environments.
        dynamics_fn: Override dynamics (default: ``two_link_arm_dynamics``).
        max_torque: Symmetric torque bound per joint.
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
        max_torque: float = 10.0,
        dt: float = 0.02,
        alpha: float = 0.01,
        lipschitz: float = 5.0,
        eval_weights: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        dynamics_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dynamics_fn is None:
            dynamics_fn = two_link_arm_dynamics

        state_spec = StateSpec(dim=4, wrap_dims=(0, 1))
        control_spec = ControlSpec(
            dim=2,
            lower_bounds=torch.tensor([-max_torque, -max_torque]),
            upper_bounds=torch.tensor([max_torque, max_torque]),
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
