"""Cart-pole dynamics."""

from __future__ import annotations

import torch


def cartpole_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    m_c: float = 1.0,
    m_p: float = 0.1,
    l: float = 0.5,
    g: float = 9.81,
) -> torch.Tensor:
    """Batched cart-pole dynamics.

    Args:
        x: States of shape ``(B, 4)`` — ``[x, x_dot, theta, theta_dot]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)`` — horizontal force.
        m_c: Cart mass.
        m_p: Pole mass.
        l: Half-pole length (to center of mass).
        g: Gravitational acceleration.

    Returns:
        State derivatives of shape ``(B, 4)``.
    """
    pos_dot = x[:, 1]
    theta = x[:, 2]
    theta_dot = x[:, 3]
    F = u.squeeze(-1)

    sin_th = torch.sin(theta)
    cos_th = torch.cos(theta)
    total_mass = m_c + m_p

    theta_ddot = (
        g * sin_th
        - cos_th * (F + m_p * l * theta_dot**2 * sin_th) / total_mass
    ) / (l * (4.0 / 3.0 - m_p * cos_th**2 / total_mass))

    x_ddot = (
        F + m_p * l * (theta_dot**2 * sin_th - theta_ddot * cos_th)
    ) / total_mass

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = pos_dot
    dxdt[:, 1] = x_ddot
    dxdt[:, 2] = theta_dot
    dxdt[:, 3] = theta_ddot
    return dxdt
