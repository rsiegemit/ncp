"""Van der Pol oscillator dynamics."""

from __future__ import annotations

import torch


def van_der_pol_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    mu: float = 1.0,
) -> torch.Tensor:
    """Batched Van der Pol oscillator dynamics.

    Args:
        x: States of shape ``(B, 2)`` — ``[x, x_dot]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)`` — external force.
        mu: Nonlinearity parameter.

    Returns:
        State derivatives of shape ``(B, 2)``.
    """
    x0 = x[:, 0]
    x1 = x[:, 1]
    force = u.squeeze(-1)

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = x1
    dxdt[:, 1] = mu * (1.0 - x0**2) * x1 - x0 + force
    return dxdt
