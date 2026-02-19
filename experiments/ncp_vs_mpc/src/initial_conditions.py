"""Initial condition generation: grid + random, shared across controllers."""

import math
import torch
import numpy as np
from pathlib import Path
from typing import Sequence


def generate_grid_ics(
    state_dim: int,
    radius: float,
    grid_per_dim: int,
    wrap_dims: tuple[int, ...] = (),
) -> torch.Tensor:
    """Generate uniform grid of initial conditions in [-R, R]^d.

    Angular dimensions (wrap_dims) are clipped to [-pi, pi].
    """
    linspaces = []
    for d in range(state_dim):
        lo, hi = -radius, radius
        if d in wrap_dims:
            lo = max(lo, -math.pi)
            hi = min(hi, math.pi)
        linspaces.append(torch.linspace(lo, hi, grid_per_dim))

    grids = torch.meshgrid(*linspaces, indexing="ij")
    points = torch.stack([g.flatten() for g in grids], dim=-1)

    # Remove the origin (or near-origin) — it's the target
    norms = points.abs().max(dim=-1).values
    mask = norms > 1e-8
    return points[mask]


def generate_random_ics(
    state_dim: int,
    radius: float,
    num_random: int,
    wrap_dims: tuple[int, ...] = (),
    seed: int = 42,
) -> torch.Tensor:
    """Generate random initial conditions uniformly in [-R, R]^d."""
    rng = torch.Generator().manual_seed(seed)
    points = (2 * torch.rand(num_random, state_dim, generator=rng) - 1) * radius

    # Clip angular dims
    for d in wrap_dims:
        points[:, d] = points[:, d].clamp(-math.pi, math.pi)

    return points


def generate_ics(
    state_dim: int,
    radius: float,
    grid_per_dim: int,
    num_random: int,
    wrap_dims: tuple[int, ...] = (),
    seed: int = 42,
) -> torch.Tensor:
    """Generate combined grid + random ICs."""
    grid = generate_grid_ics(state_dim, radius, grid_per_dim, wrap_dims)
    rand = generate_random_ics(state_dim, radius, num_random, wrap_dims, seed)
    return torch.cat([grid, rand], dim=0)


def save_ics(ics: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ics, path)


def load_ics(path: Path) -> torch.Tensor:
    return torch.load(path, weights_only=True)
