"""Duffing oscillator dynamics."""

from __future__ import annotations

import torch


def duffing_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    delta: float = 0.3,
) -> torch.Tensor:
    """Batched Duffing oscillator dynamics.

    Args:
        x: States of shape ``(B, 2)`` — ``[x, x_dot]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)`` — external force.
        alpha: Linear stiffness.
        beta: Cubic stiffness.
        delta: Damping coefficient.

    Returns:
        State derivatives of shape ``(B, 2)``.
    """
    x0 = x[:, 0]
    x1 = x[:, 1]
    force = u.squeeze(-1)

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = x1
    dxdt[:, 1] = -delta * x1 - alpha * x0 - beta * x0**3 + force
    return dxdt
