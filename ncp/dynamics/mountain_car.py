"""Mountain car dynamics."""

from __future__ import annotations

import torch


def mountain_car_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Batched continuous mountain car dynamics.

    Args:
        x: States of shape ``(B, 2)`` — ``[position, velocity]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)`` — force.

    Returns:
        State derivatives of shape ``(B, 2)``.
    """
    velocity = x[:, 1]
    force = u.squeeze(-1)

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = velocity
    dxdt[:, 1] = force - 0.0025 * torch.cos(3.0 * x[:, 0])
    return dxdt
