"""Point-in-hypercube lookup."""

from __future__ import annotations

import torch


def find_hypercubes(
    points: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Find which hypercube (if any) each point belongs to.

    Args:
        points: ``(k, D)`` tensor of query points.
        centers: ``(m, D)`` tensor of hypercube centres.
        radii: ``(m,)`` tensor of half-widths.

    Returns:
        ``(k,)`` integer tensor of hypercube indices, or ``-1`` if a point
        is not contained in any hypercube.
    """
    # (k, 1, D) vs (1, m, D)
    points_exp = points[:, None, :]
    centers_exp = centers[None, :, :]
    radii_exp = radii[None, :, None]

    lower = centers_exp - radii_exp
    upper = centers_exp + radii_exp

    contained = (points_exp >= lower) & (points_exp <= upper)
    contained = contained.all(dim=-1)  # (k, m)

    any_match = contained.any(dim=1)
    match_idx = torch.argmax(contained.int(), dim=1)

    return torch.where(any_match, match_idx, torch.full_like(match_idx, -1))
