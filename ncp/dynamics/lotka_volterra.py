"""Lotka-Volterra (predator-prey) dynamics."""

from __future__ import annotations

import torch


def lotka_volterra_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    a: float = 1.5,
    b: float = 1.0,
    c: float = 3.0,
    d: float = 1.0,
) -> torch.Tensor:
    """Batched Lotka-Volterra dynamics with harvesting control.

    Args:
        x: States of shape ``(B, 2)`` — ``[prey, predator]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)`` — prey harvesting rate.
        a: Prey growth rate.
        b: Predation rate.
        c: Predator death rate.
        d: Predator reproduction rate.

    Returns:
        State derivatives of shape ``(B, 2)``.
    """
    prey = x[:, 0]
    predator = x[:, 1]
    harvest = u.squeeze(-1)

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = a * prey - b * prey * predator - harvest * prey
    dxdt[:, 1] = d * prey * predator - c * predator
    return dxdt
