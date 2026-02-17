"""Acrobot dynamics (Spong formulation)."""

from __future__ import annotations

import torch


def acrobot_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    lc1: float = 0.5,
    lc2: float = 0.5,
    I1: float = 1.0,
    I2: float = 1.0,
    g: float = 9.81,
) -> torch.Tensor:
    """Batched acrobot dynamics (torque on second joint only).

    Args:
        x: States of shape ``(B, 4)`` — ``[q1, q2, q1_dot, q2_dot]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)`` — torque on joint 2.
        m1: Mass of link 1.
        m2: Mass of link 2.
        l1: Length of link 1.
        l2: Length of link 2.
        lc1: Distance to center of mass of link 1.
        lc2: Distance to center of mass of link 2.
        I1: Moment of inertia of link 1.
        I2: Moment of inertia of link 2.
        g: Gravitational acceleration.

    Returns:
        State derivatives of shape ``(B, 4)``.
    """
    q1 = x[:, 0]
    q2 = x[:, 1]
    q1_dot = x[:, 2]
    q2_dot = x[:, 3]
    tau2 = u.squeeze(-1)

    c2 = torch.cos(q2)
    s2 = torch.sin(q2)
    s1 = torch.sin(q1)
    s12 = torch.sin(q1 + q2)

    h = m2 * l1 * lc2
    d11 = I1 + I2 + m2 * l1**2 + 2.0 * h * c2
    d12 = I2 + h * c2
    d22 = torch.tensor(I2, dtype=x.dtype, device=x.device)

    # Coriolis
    c1 = -h * s2 * q2_dot * (2.0 * q1_dot + q2_dot)
    c2_term = h * s2 * q1_dot**2

    # Gravity
    g1 = (m1 * lc1 + m2 * l1) * g * s1 + m2 * g * lc2 * s12
    g2 = m2 * g * lc2 * s12

    # Solve M * q_ddot = tau - C - G, with B = [0; 1]
    rhs1 = -c1 - g1
    rhs2 = tau2 - c2_term - g2

    det = d11 * d22 - d12 * d12
    q1_ddot = (d22 * rhs1 - d12 * rhs2) / det
    q2_ddot = (-d12 * rhs1 + d11 * rhs2) / det

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = q1_dot
    dxdt[:, 1] = q2_dot
    dxdt[:, 2] = q1_ddot
    dxdt[:, 3] = q2_ddot
    return dxdt
