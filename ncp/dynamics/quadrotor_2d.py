"""Planar quadrotor (PVTOL) dynamics."""

from __future__ import annotations

import torch


def quadrotor_2d_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    m: float = 1.0,
    J: float = 0.01,
    d: float = 0.2,
    g: float = 9.81,
) -> torch.Tensor:
    """Batched planar quadrotor dynamics.

    Args:
        x: States of shape ``(B, 6)`` —
            ``[x, y, x_dot, y_dot, phi, phi_dot]``.
        u: Controls of shape ``(B, 2)`` — ``[f1, f2]`` (left/right thrust).
        m: Quadrotor mass.
        J: Moment of inertia.
        d: Half-distance between rotors.
        g: Gravitational acceleration.

    Returns:
        State derivatives of shape ``(B, 6)``.
    """
    phi = x[:, 4]
    phi_dot = x[:, 5]
    f1 = u[:, 0]
    f2 = u[:, 1]

    total_thrust = f1 + f2
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = x[:, 2]
    dxdt[:, 1] = x[:, 3]
    dxdt[:, 2] = -total_thrust * sin_phi / m
    dxdt[:, 3] = total_thrust * cos_phi / m - g
    dxdt[:, 4] = phi_dot
    dxdt[:, 5] = (f1 - f2) * d / J
    return dxdt
