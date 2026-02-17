"""Tests for dynamics modules."""

import torch
import pytest

from ncp.dynamics.pendulum import (
    inverted_pendulum_2d_torch,
    simplified_pendulum_derivatives,
)
from ncp.dynamics.unicycle import unicycle_derivatives, unicycle_derivatives_radial
from ncp.dynamics.bicycle import bicycle_derivatives


class TestPendulumDynamics:
    def test_inverted_pendulum_shape(self) -> None:
        x = torch.randn(10, 2)
        u = torch.randn(10)
        dx = inverted_pendulum_2d_torch(x, u)
        assert dx.shape == (10, 2)

    def test_inverted_pendulum_at_equilibrium(self) -> None:
        x = torch.zeros(1, 2)
        u = torch.zeros(1)
        dx = inverted_pendulum_2d_torch(x, u)
        assert torch.allclose(dx, torch.zeros(1, 2), atol=1e-6)

    def test_simplified_pendulum_shape(self) -> None:
        x = torch.randn(5, 2)
        u = torch.randn(5)
        dx = simplified_pendulum_derivatives(x, u)
        assert dx.shape == (5, 2)

    def test_simplified_pendulum_first_component_is_velocity(self) -> None:
        x = torch.tensor([[0.5, 1.0]])
        u = torch.zeros(1)
        dx = simplified_pendulum_derivatives(x, u)
        assert torch.isclose(dx[0, 0], x[0, 1])


class TestUnicycleDynamics:
    def test_unicycle_shape(self) -> None:
        x = torch.randn(8, 3)
        u = torch.randn(8, 2)
        dx = unicycle_derivatives(x, u)
        assert dx.shape == (8, 3)

    def test_unicycle_straight_line(self) -> None:
        # theta=0, v=1, omega=0 => dx=1, dy=0, dtheta=0
        x = torch.tensor([[0.0, 0.0, 0.0]])
        u = torch.tensor([[1.0, 0.0]])
        dx = unicycle_derivatives(x, u)
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(dx, expected, atol=1e-6)

    def test_unicycle_with_wind(self) -> None:
        import math
        x = torch.tensor([[0.0, 5.5 * math.pi, 0.0]])
        u = torch.tensor([[1.0, 0.0]])
        dx_no_wind = unicycle_derivatives(x, u, wind=False)
        dx_wind = unicycle_derivatives(x, u, wind=True, windval=-0.5)
        # Wind should reduce dx/dt
        assert dx_wind[0, 0] < dx_no_wind[0, 0]

    def test_radial_shape(self) -> None:
        x = torch.randn(5, 3)
        u = torch.randn(5, 2)
        dx = unicycle_derivatives_radial(x, u)
        assert dx.shape == (5, 3)


class TestBicycleDynamics:
    def test_bicycle_shape(self) -> None:
        x = torch.randn(4, 3)
        u = torch.randn(4, 2)
        dx = bicycle_derivatives(x, u)
        assert dx.shape == (4, 3)

    def test_bicycle_straight_line(self) -> None:
        x = torch.tensor([[0.0, 0.0, 0.0]])
        u = torch.tensor([[1.0, 0.0]])  # v=1, delta=0
        dx = bicycle_derivatives(x, u)
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(dx, expected, atol=1e-6)
