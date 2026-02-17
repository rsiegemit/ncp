"""Alpha search and certification for NCP contraction rates."""

from __future__ import annotations

import torch


@torch.no_grad()
def search_alpha_parallel(
    sol: torch.Tensor,
    v_start: torch.Tensor,
    radius: torch.Tensor,
    t_eval: torch.Tensor,
    lipschitz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary-search for the tightest contraction rate *alpha* per cell.

    Args:
        sol: ``(B, T+1)`` trajectory norms.
        v_start: ``(B,)`` initial Lyapunov values.
        radius: ``(B,)`` cell half-widths.
        t_eval: ``(T+1,)`` evaluation time grid.
        lipschitz: Lipschitz constant *L*.

    Returns:
        ``(alpha, indices)`` — contraction rates and minimising time indices.
    """
    if sol.dim() > 1:
        sol = sol[:, 1:]
    else:
        sol = sol[1:]
    t_eval = t_eval[1:]

    alpha_h = torch.ones_like(v_start)
    alpha_l = -torch.ones_like(v_start)
    threshold = v_start - radius

    # Handle the edge case where V_start <= radius
    safe_mask = threshold > 0

    if sol.dim() > 1:
        r = radius.unsqueeze(1).expand_as(sol)
    else:
        r = radius.unsqueeze(0).expand_as(sol)

    # Pre-compute lipschitz * t_eval (reused in every _compute_val)
    L_t = lipschitz * t_eval

    def _compute_val(alpha: torch.Tensor) -> torch.Tensor:
        alpha_t = torch.outer(alpha, t_eval)
        return torch.min(
            sol * torch.exp(alpha_t) + r * torch.exp(alpha_t + L_t),
            dim=1,
        ).values

    condition_h = _compute_val(alpha_h) > threshold
    condition_l = _compute_val(alpha_l) < threshold

    while torch.any((~condition_h) & safe_mask) or torch.any(
        (~condition_l) & safe_mask
    ):
        alpha_h[~condition_h] *= 2
        alpha_l[~condition_l] *= 2
        condition_h = _compute_val(alpha_h) > threshold
        condition_l = _compute_val(alpha_l) < threshold

    # For cells where V_start <= radius, collapse the search interval
    alpha_l[~safe_mask] = alpha_h[~safe_mask]

    # Fixed 14 iterations of bisection (precision ~6e-5, sufficient for 1e-4)
    for _ in range(14):
        alpha_mid = (alpha_h + alpha_l) * 0.5
        condition = _compute_val(alpha_mid) < threshold
        alpha_l = torch.where(condition, alpha_mid, alpha_l)
        alpha_h = torch.where(condition, alpha_h, alpha_mid)

    alpha_mid = (alpha_h + alpha_l) * 0.5

    # Compute final argmin
    alpha_t = torch.outer(alpha_mid, t_eval)
    combined = sol * torch.exp(alpha_t) + r * torch.exp(alpha_t + L_t)
    indices = combined.argmin(dim=1)
    return alpha_mid, indices


@torch.no_grad()
def certify_alpha(
    sol: torch.Tensor,
    v_start: torch.Tensor,
    radius: torch.Tensor,
    t_eval: torch.Tensor,
    lipschitz: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Certify that a given *alpha* satisfies the contraction condition.

    Args:
        sol: ``(B, T+1)`` trajectory norms.
        v_start: ``(B,)`` initial Lyapunov values.
        radius: ``(B,)`` cell half-widths.
        t_eval: ``(T+1,)`` evaluation time grid.
        lipschitz: Lipschitz constant *L*.
        alpha: Contraction rate to certify.

    Returns:
        ``(condition, indices)`` — boolean mask of passing cells
        and minimising time indices.
    """
    sol = sol[:, 1:]
    t_eval = t_eval[1:]

    alpha_t = alpha * t_eval
    r = radius.unsqueeze(1).expand_as(sol)

    val = sol * torch.exp(alpha_t) + r * torch.exp(alpha_t + lipschitz * t_eval)
    min_result = val.min(dim=1)
    condition = min_result.values < v_start - radius
    return condition, min_result.indices
