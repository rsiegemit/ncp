"""Tests for geometry modules."""

import math

import torch
import pytest

from ncp.geometry.hypercubes import find_hypercubes
from ncp.geometry.intersections import (
    count_hypercube_intersections,
    find_hypercube_intersections,
)
from ncp.geometry.grid import generate_initial_grid, subdivide_cells


class TestFindHypercubes:
    def test_point_inside(self) -> None:
        centers = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        radii = torch.tensor([0.5, 0.5])
        points = torch.tensor([[0.1, 0.1]])
        result = find_hypercubes(points, centers, radii)
        assert result.item() == 0

    def test_point_outside(self) -> None:
        centers = torch.tensor([[0.0, 0.0]])
        radii = torch.tensor([0.5])
        points = torch.tensor([[5.0, 5.0]])
        result = find_hypercubes(points, centers, radii)
        assert result.item() == -1

    def test_batch(self) -> None:
        centers = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        radii = torch.tensor([0.5, 0.5])
        points = torch.tensor([[0.1, 0.1], [2.1, 2.1], [10.0, 10.0]])
        result = find_hypercubes(points, centers, radii)
        assert result.tolist() == [0, 1, -1]


class TestIntersections:
    def test_count_no_wrap(self) -> None:
        centers = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
        radii = torch.tensor([0.5, 0.5])
        query_centers = torch.tensor([[0.3, 0.3]])
        query_radii = torch.tensor([0.1])
        counts = count_hypercube_intersections(
            centers, radii, query_centers, query_radii
        )
        assert counts.item() == 2

    def test_count_with_wrap(self) -> None:
        # Two boxes near the wrapping boundary
        centers = torch.tensor([[3.0, 0.0], [-3.0, 0.0]])
        radii = torch.tensor([0.3, 0.3])
        query_centers = torch.tensor([[3.1, 0.0]])
        query_radii = torch.tensor([0.3])
        # Without wrap, only first box intersects
        count_no_wrap = count_hypercube_intersections(
            centers, radii, query_centers, query_radii, wrap_dims=()
        )
        # With wrap on dim 0, the second box at -3.0 should also be close
        count_wrap = count_hypercube_intersections(
            centers, radii, query_centers, query_radii, wrap_dims=(0,)
        )
        assert count_no_wrap.item() == 1
        assert count_wrap.item() == 2

    def test_find_returns_indices(self) -> None:
        centers = torch.tensor([[0.0, 0.0], [0.5, 0.0], [5.0, 5.0]])
        radii = torch.tensor([0.5, 0.5, 0.5])
        query_centers = torch.tensor([[0.3, 0.0]])
        query_radii = torch.tensor([0.1])
        result = find_hypercube_intersections(
            centers, radii, query_centers, query_radii
        )
        assert len(result) == 1
        indices = sorted(result[0].tolist())
        assert indices == [0, 1]


class TestGrid:
    def test_initial_grid_2d(self) -> None:
        points, radii = generate_initial_grid(2, 0.1, 1.0)
        assert points.dim() == 2
        assert points.shape[1] == 2
        assert radii.shape[0] == points.shape[0]
        # No point at origin
        norms = torch.linalg.norm(points, dim=1)
        assert norms.min() > 0

    def test_initial_grid_3d(self) -> None:
        points, radii = generate_initial_grid(3, 0.1, 1.0)
        assert points.shape[1] == 3

    def test_subdivide_2d(self) -> None:
        centers = torch.tensor([[1.0, 1.0]])
        radii = torch.tensor([0.3])
        new_c, new_r = subdivide_cells(centers, radii, 2)
        assert new_c.shape == (9, 2)
        assert new_r.shape == (9,)
        assert torch.allclose(new_r, torch.tensor(0.1).expand(9), atol=1e-6)
