"""Dimension-agnostic grid generation and subdivision."""

from __future__ import annotations

import math

import torch


def generate_initial_grid(
    d: int,
    epsilon: float,
    R: float,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the multi-resolution starting grid used by the NCP algorithm.

    Creates a set of hypercube centres and associated half-widths
    covering ``[-R, R]^d`` (excluding the origin cell) at geometrically
    increasing resolution layers.

    Args:
        d: State-space dimension.
        epsilon: Finest resolution half-width.
        R: Outer radius of the region to cover.
        device: Torch device.

    Returns:
        ``(points, radii)`` — centres ``(N, d)`` and half-widths ``(N,)``.
    """
    if device is None:
        device = torch.device("cpu")

    num_layers = math.ceil(math.log(R / epsilon, 3))

    # Offsets per layer: {-2*eps*3^i, 0, 2*eps*3^i}
    offsets = torch.tensor(
        [round(2 * epsilon * 3**i, 8) for i in range(num_layers)],
        device=device,
    )
    per_dim_offsets = torch.stack([-offsets, torch.zeros_like(offsets), offsets], dim=1)
    # per_dim_offsets: (num_layers, 3)

    # Cartesian product across d dimensions
    layer_points_list: list[torch.Tensor] = []
    layer_radii_list: list[torch.Tensor] = []

    for layer_idx in range(num_layers):
        layer_vals = per_dim_offsets[layer_idx]  # (3,)
        # Generate d-dimensional Cartesian product
        grids = [layer_vals] * d
        combos = torch.cartesian_prod(*grids) if d > 1 else layer_vals.unsqueeze(1)
        # Exclude the origin
        nonzero_mask = combos.abs().sum(dim=-1) > 0
        combos = combos[nonzero_mask]

        half_width = round(epsilon * 3**layer_idx, 8)
        layer_points_list.append(combos)
        layer_radii_list.append(
            torch.full((combos.shape[0],), half_width, device=device)
        )

    points = torch.cat(layer_points_list, dim=0).to(device)
    radii = torch.cat(layer_radii_list, dim=0).to(device)

    return points, radii


def subdivide_cells(
    centers: torch.Tensor,
    radii: torch.Tensor,
    d: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subdivide failed cells into 3^d sub-cells.

    Each cell is split into a uniform 3x3x...x3 grid of smaller cells.

    Args:
        centers: ``(N, d)`` centres of cells to subdivide.
        radii: ``(N,)`` half-widths of cells to subdivide.
        d: State-space dimension.

    Returns:
        ``(new_centers, new_radii)`` — the subdivided cells.
    """
    device = centers.device

    # Build offset combinations: (-2/3*r, 0, 2/3*r) for each dim
    offset_fracs = torch.tensor([-2.0 / 3, 0.0, 2.0 / 3], device=device)

    # For each parent cell, create offsets scaled by its radius
    # offset_combos shape: (3^d, d)
    grids = [offset_fracs] * d
    offset_combos = torch.cartesian_prod(*grids) if d > 1 else offset_fracs.unsqueeze(1)

    n_sub = offset_combos.shape[0]  # 3^d

    # Scale offsets by each cell's radius and add to centres
    # radii: (N,) -> (N, 1, 1) for broadcasting
    scaled_offsets = offset_combos.unsqueeze(0) * radii[:, None, None]  # (N, 3^d, d)
    new_centers = centers.unsqueeze(1) + scaled_offsets  # (N, 3^d, d)
    new_centers = new_centers.reshape(-1, d)

    new_radii = (radii / 3.0).repeat_interleave(n_sub)

    return new_centers, new_radii
