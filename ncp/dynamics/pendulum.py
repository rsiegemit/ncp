"""Inverted pendulum dynamics."""

from __future__ import annotations

import torch


def inverted_pendulum_2d_torch(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    m: float = 0.1,
    l: float = 10.0,
    g: float = 9.81,
) -> torch.Tensor:
    """2-D simplified inverted pendulum dynamics (batched).

    Args:
        x: States of shape ``(B, 2)`` — ``[theta, theta_dot]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)``.
        m: Pendulum mass.
        l: Length to center of mass.
        g: Gravitational acceleration.

    Returns:
        State derivatives of shape ``(B, 2)`` — ``[theta_dot, theta_ddot]``.
    """
    theta = x[:, 0]
    theta_dot = x[:, 1]
    u = u.squeeze(-1)

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = theta_dot
    dxdt[:, 1] = (g / l) * torch.sin(theta) + (u / (m * l)) * torch.abs(
        torch.cos(theta)
    )
    return dxdt


def simplified_pendulum_derivatives(
    x: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Pendulum-v1 dynamics as implemented in OpenAI Gym.

    Args:
        x: States of shape ``(B, 2)`` — ``[theta, theta_dot]``.
        u: Controls of shape ``(B,)`` — torque in ``[-2, 2]``.

    Returns:
        State derivatives of shape ``(B, 2)``.
    """
    theta = x[:, 0]
    theta_dot = x[:, 1]

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = theta_dot
    dxdt[:, 1] = 30.0 * torch.sin(theta) + 15.0 * u
    return dxdt
