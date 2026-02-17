"""Hypercube intersection queries with configurable wrapping dimensions."""

from __future__ import annotations

import math

import torch


def _expand_radii(radii: torch.Tensor, n: int, d: int) -> torch.Tensor:
    """Ensure radii has shape ``(n, d)``.

    Accepts:
    - ``(n,)`` — one scalar radius per hypercube, broadcast to all dims.
    - ``(n, 1)`` — same, explicit column.
    - ``(n, d)`` — per-dimension radii, used as-is.
    """
    if radii.ndim == 1:
        if radii.shape[0] == n:
            return radii.unsqueeze(1).expand(n, d)
        if radii.shape[0] == d and n == 1:
            # Treat as per-dimension radii for a single box
            return radii.unsqueeze(0)
        return radii.unsqueeze(1).expand(n, d)
    if radii.shape[1] == 1:
        return radii.expand(n, d)
    return radii


def _build_intersection_mask(
    centers: torch.Tensor,
    radii: torch.Tensor,
    query_centers: torch.Tensor,
    query_radii: torch.Tensor,
    wrap_dims: tuple[int, ...] = (),
) -> torch.Tensor:
    """Return a ``(M, N)`` boolean mask where entry ``(i, j)`` is True if
    query hypercube *i* intersects stored hypercube *j*.

    Args:
        centers: ``(N, D)`` stored hypercube centres.
        radii: ``(N,)`` or ``(N, D)`` stored half-widths.
        query_centers: ``(M, D)`` query centres.
        query_radii: ``(M,)`` or ``(M, D)`` query half-widths.
        wrap_dims: Tuple of dimension indices that wrap at ``2*pi``.
    """
    n, d = centers.shape
    m = query_centers.shape[0]

    radii = _expand_radii(radii, n, d)
    query_radii = _expand_radii(query_radii, m, d)

    # Broadcast: (1, N, D) and (M, 1, D)
    c = centers.unsqueeze(0)
    r = radii.unsqueeze(0)
    qc = query_centers.unsqueeze(1)
    qr = query_radii.unsqueeze(1)

    # Per-dimension distance
    raw_diff = torch.abs(c - qc)  # (M, N, D)

    # For wrapped dimensions, use circular distance
    if wrap_dims:
        wrap_idx = list(wrap_dims)
        wrapped_diff = torch.minimum(
            raw_diff[..., wrap_idx],
            2 * math.pi - raw_diff[..., wrap_idx],
        )
        raw_diff = raw_diff.clone()
        raw_diff[..., wrap_idx] = wrapped_diff

    threshold = r + qr  # (M, N, D)
    return (raw_diff <= threshold).all(dim=-1)  # (M, N)


def count_hypercube_intersections(
    centers: torch.Tensor,
    radii: torch.Tensor,
    query_centers: torch.Tensor,
    query_radii: torch.Tensor,
    wrap_dims: tuple[int, ...] = (),
) -> torch.Tensor:
    """Count how many stored hypercubes intersect each query hypercube.

    Args:
        centers: ``(N, D)`` stored centres.
        radii: ``(N,)`` or ``(N, D)`` stored half-widths.
        query_centers: ``(M, D)`` query centres.
        query_radii: ``(M,)`` or ``(M, D)`` query half-widths.
        wrap_dims: Dimension indices that wrap at ``2*pi``.

    Returns:
        ``(M,)`` integer tensor of intersection counts.
    """
    mask = _build_intersection_mask(
        centers, radii, query_centers, query_radii, wrap_dims
    )
    return mask.sum(dim=1)


def find_hypercube_intersections(
    centers: torch.Tensor,
    radii: torch.Tensor,
    query_centers: torch.Tensor,
    query_radii: torch.Tensor,
    wrap_dims: tuple[int, ...] = (),
) -> list[torch.Tensor]:
    """For each query, return indices of intersecting stored hypercubes.

    Args:
        centers: ``(N, D)`` stored centres.
        radii: ``(N,)`` or ``(N, D)`` stored half-widths.
        query_centers: ``(M, D)`` query centres.
        query_radii: ``(M,)`` or ``(M, D)`` query half-widths.
        wrap_dims: Dimension indices that wrap at ``2*pi``.

    Returns:
        A list of length ``M`` where each element is a 1-D tensor of
        intersecting hypercube indices.
    """
    mask = _build_intersection_mask(
        centers, radii, query_centers, query_radii, wrap_dims
    )
    m = query_centers.shape[0]

    query_idx, box_idx = torch.nonzero(mask, as_tuple=True)
    counts = torch.bincount(query_idx, minlength=m)
    splits = counts.cumsum(0)
    splits = torch.cat([splits.new_zeros(1), splits])

    return [box_idx[splits[i] : splits[i + 1]] for i in range(m)]
