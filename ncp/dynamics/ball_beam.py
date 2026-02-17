"""Ball and beam dynamics."""

from __future__ import annotations

import torch


def ball_beam_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    m: float = 0.05,
    J: float = 0.02,
    g: float = 9.81,
) -> torch.Tensor:
    """Batched ball-and-beam dynamics.

    Args:
        x: States of shape ``(B, 4)`` — ``[r, r_dot, theta, theta_dot]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)`` — beam torque.
        m: Ball mass.
        J: Beam moment of inertia.
        g: Gravitational acceleration.

    Returns:
        State derivatives of shape ``(B, 4)``.
    """
    r = x[:, 0]
    r_dot = x[:, 1]
    theta = x[:, 2]
    theta_dot = x[:, 3]
    tau = u.squeeze(-1)

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = r_dot
    dxdt[:, 1] = r * theta_dot**2 - g * torch.sin(theta)
    dxdt[:, 2] = theta_dot
    dxdt[:, 3] = (
        tau - 2.0 * m * r * r_dot * theta_dot - m * g * r * torch.cos(theta)
    ) / (m * r**2 + J)
    return dxdt
