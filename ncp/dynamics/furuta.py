"""Furuta pendulum (rotary inverted pendulum) dynamics."""

from __future__ import annotations

import torch


def furuta_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    m_arm: float = 0.095,
    m_pend: float = 0.024,
    L_arm: float = 0.085,
    l_pend: float = 0.129,
    J_arm: float = 5.72e-5,
    J_pend: float = 3.33e-5,
    g: float = 9.81,
) -> torch.Tensor:
    """Batched Furuta pendulum dynamics (Quanser-like parameters).

    Args:
        x: States of shape ``(B, 4)`` —
            ``[theta_arm, theta_pend, dtheta_arm, dtheta_pend]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)`` — motor torque.
        m_arm: Arm mass.
        m_pend: Pendulum mass.
        L_arm: Arm length.
        l_pend: Distance to pendulum center of mass.
        J_arm: Arm moment of inertia.
        J_pend: Pendulum moment of inertia.
        g: Gravitational acceleration.

    Returns:
        State derivatives of shape ``(B, 4)``.
    """
    th_a_dot = x[:, 2]
    th_p = x[:, 1]
    th_p_dot = x[:, 3]
    tau = u.squeeze(-1)

    cp = torch.cos(th_p)
    sp = torch.sin(th_p)

    a = J_arm + m_pend * L_arm**2
    b = m_pend * L_arm * l_pend
    d = J_pend + m_pend * l_pend**2

    det = a * d - (b * cp) ** 2

    # Coriolis and gravity terms
    c1 = -2.0 * b * sp * cp * th_a_dot * th_p_dot - b * sp * th_p_dot**2
    c2 = b * sp * cp * th_a_dot**2
    g2 = m_pend * g * l_pend * sp

    rhs1 = tau + c1
    rhs2 = -c2 - g2

    th_a_ddot = (d * rhs1 - b * cp * rhs2) / det
    th_p_ddot = (-b * cp * rhs1 + a * rhs2) / det

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = th_a_dot
    dxdt[:, 1] = th_p_dot
    dxdt[:, 2] = th_a_ddot
    dxdt[:, 3] = th_p_ddot
    return dxdt
