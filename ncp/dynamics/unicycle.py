"""Unicycle dynamics."""

from __future__ import annotations

import math

import torch

_5PI = 5 * math.pi
_6PI = 6 * math.pi


def unicycle_derivatives(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    wind: bool = False,
    windval: float = -0.6,
    windbot: float = _5PI,
    windtop: float = _6PI,
) -> torch.Tensor:
    """Continuous-time unicycle model.

    Args:
        x: States of shape ``(B, 3)`` — ``[x, y, theta]``.
        u: Controls of shape ``(B, 2)`` — ``[v, omega]``.
        wind: Whether to apply a wind perturbation.
        windval: Magnitude of the wind disturbance on ``dx/dt``.
        windbot: Lower y-bound of the wind region.
        windtop: Upper y-bound of the wind region.

    Returns:
        State derivatives of shape ``(B, 3)``.
    """
    theta = x[:, 2]
    v = u[:, 0]

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = v * torch.cos(theta)
    if wind:
        mask = (x[:, 1] >= windbot) & (x[:, 1] <= windtop)
        dxdt[:, 0] = dxdt[:, 0].clone()
        dxdt[:, 0][mask] += windval
    dxdt[:, 1] = v * torch.sin(theta)
    dxdt[:, 2] = u[:, 1]
    return dxdt


def unicycle_derivatives_radial(
    x: torch.Tensor,
    u: torch.Tensor,
    **_kwargs: object,
) -> torch.Tensor:
    """Unicycle model with radial repulsion when distance > 1.

    Args:
        x: States of shape ``(B, 3)`` — ``[x, y, theta]``.
        u: Controls of shape ``(B, 2)`` — ``[v, omega]``.

    Returns:
        State derivatives of shape ``(B, 3)``.
    """
    theta = x[:, 2]
    v = u[:, 0]

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = v * torch.cos(theta)
    dxdt[:, 1] = v * torch.sin(theta)
    dxdt[:, 2] = u[:, 1]

    r = torch.linalg.norm(x[:, :2], dim=1)
    mask = r > 1
    if mask.any():
        r_masked = r[mask]
        xy_masked = x[mask, :2]
        repulsion_mag = -(1.0 - r_masked) / (r_masked + 2) ** 2
        repulsion_dir = xy_masked / r_masked.unsqueeze(1)
        repulsion = repulsion_mag.unsqueeze(1) * repulsion_dir
        dxdt[mask, 0] += repulsion[:, 0]
        dxdt[mask, 1] += repulsion[:, 1]

    return dxdt
