"""Continuous Stirred Tank Reactor (CSTR) dynamics."""

from __future__ import annotations

import torch


def cstr_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    F: float = 1.0,
    V: float = 1.0,
    C_Af: float = 10.0,
    T_f: float = 300.0,
    k0: float = 7.2e10,
    E_over_R: float = 8750.0,
    dH: float = -5e4,
    rho: float = 1000.0,
    Cp: float = 0.239,
) -> torch.Tensor:
    """Batched CSTR dynamics.

    Args:
        x: States of shape ``(B, 2)`` — ``[C_A, T]``.
        u: Controls of shape ``(B,)`` or ``(B, 1)`` — heat input ``Q``.
        F: Volumetric flow rate.
        V: Reactor volume.
        C_Af: Feed concentration.
        T_f: Feed temperature.
        k0: Pre-exponential factor.
        E_over_R: Activation energy divided by gas constant.
        dH: Heat of reaction.
        rho: Fluid density.
        Cp: Heat capacity.

    Returns:
        State derivatives of shape ``(B, 2)``.
    """
    C_A = x[:, 0]
    T = x[:, 1]
    Q = u.squeeze(-1)

    rate = k0 * torch.exp(-E_over_R / T) * C_A

    dxdt = torch.empty_like(x)
    dxdt[:, 0] = (F / V) * (C_Af - C_A) - rate
    dxdt[:, 1] = (
        (F / V) * (T_f - T)
        + (-dH / (rho * Cp)) * rate
        + Q / (rho * Cp * V)
    )
    return dxdt
