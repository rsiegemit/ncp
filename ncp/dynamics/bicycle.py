"""Kinematic bicycle model dynamics."""

from __future__ import annotations

import torch


def bicycle_derivatives(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    wheelbase: float = 2.0,
) -> torch.Tensor:
    """Continuous-time kinematic bicycle model.

    Args:
        x: States of shape ``(B, 3)`` — ``[x, y, theta]``.
        u: Controls of shape ``(B, 2)`` — ``[v, delta]``.
        wheelbase: Distance between front and rear axles.

    Returns:
        State derivatives of shape ``(B, 3)``.
    """
    theta = x[:, 2]
    v = u[:, 0]
    delta = u[:, 1]

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = v * torch.cos(theta)
    dxdt[:, 1] = v * torch.sin(theta)
    dxdt[:, 2] = v / wheelbase * torch.tan(delta)
    return dxdt
