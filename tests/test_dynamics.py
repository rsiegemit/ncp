"""Tests for dynamics modules."""

import torch
import pytest

from ncp.dynamics.pendulum import (
    inverted_pendulum_2d_torch,
    simplified_pendulum_derivatives,
)
from ncp.dynamics.unicycle import unicycle_derivatives, unicycle_derivatives_radial
from ncp.dynamics.bicycle import bicycle_derivatives
from ncp.dynamics.cartpole import cartpole_dynamics
from ncp.dynamics.acrobot import acrobot_dynamics
from ncp.dynamics.mountain_car import mountain_car_dynamics
from ncp.dynamics.pendubot import pendubot_dynamics
from ncp.dynamics.furuta import furuta_dynamics
from ncp.dynamics.ball_beam import ball_beam_dynamics
from ncp.dynamics.quadrotor_2d import quadrotor_2d_dynamics
from ncp.dynamics.two_link_arm import two_link_arm_dynamics
from ncp.dynamics.cstr import cstr_dynamics
from ncp.dynamics.van_der_pol import van_der_pol_dynamics
from ncp.dynamics.duffing import duffing_dynamics
from ncp.dynamics.lotka_volterra import lotka_volterra_dynamics


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


class TestCartPoleDynamics:
    def test_shape(self) -> None:
        x = torch.randn(10, 4)
        u = torch.randn(10)
        dx = cartpole_dynamics(x, u)
        assert dx.shape == (10, 4)

    def test_shape_with_u_2d(self) -> None:
        x = torch.randn(10, 4)
        u = torch.randn(10, 1)
        dx = cartpole_dynamics(x, u)
        assert dx.shape == (10, 4)

    def test_equilibrium(self) -> None:
        """At upright equilibrium (all zeros), derivatives should be zero."""
        x = torch.zeros(1, 4)
        u = torch.zeros(1)
        dx = cartpole_dynamics(x, u)
        assert torch.allclose(dx, torch.zeros(1, 4), atol=1e-6)

    def test_gravity_acts_on_displaced_pole(self) -> None:
        """Pole displaced from vertical should have nonzero theta_ddot."""
        x = torch.tensor([[0.0, 0.0, 0.3, 0.0]])
        u = torch.zeros(1)
        dx = cartpole_dynamics(x, u)
        assert dx[0, 3].abs() > 0.1  # theta_ddot should be significant

    def test_first_component_is_velocity(self) -> None:
        x = torch.tensor([[1.0, 2.5, 0.0, 0.0]])
        u = torch.zeros(1)
        dx = cartpole_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 1])


class TestAcrobotDynamics:
    def test_shape(self) -> None:
        x = torch.randn(8, 4)
        u = torch.randn(8)
        dx = acrobot_dynamics(x, u)
        assert dx.shape == (8, 4)

    def test_velocity_passthrough(self) -> None:
        """First two derivatives should be the velocities."""
        x = torch.tensor([[0.5, -0.3, 1.2, -0.8]])
        u = torch.zeros(1)
        dx = acrobot_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 2])
        assert torch.isclose(dx[0, 1], x[0, 3])

    def test_gravity_at_horizontal(self) -> None:
        """Links hanging down (q1=q2=0) should feel gravity."""
        x = torch.zeros(1, 4)
        u = torch.zeros(1)
        dx = acrobot_dynamics(x, u)
        # At (0,0) sin is 0, so accelerations should be near zero
        assert torch.allclose(dx[0, 2:], torch.zeros(2), atol=1e-5)


class TestMountainCarDynamics:
    def test_shape(self) -> None:
        x = torch.randn(6, 2)
        u = torch.randn(6)
        dx = mountain_car_dynamics(x, u)
        assert dx.shape == (6, 2)

    def test_first_component_is_velocity(self) -> None:
        x = torch.tensor([[0.0, 0.5]])
        u = torch.zeros(1)
        dx = mountain_car_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 1])

    def test_gravity_hill(self) -> None:
        """At position=0, cos(0)=1, so gravity term is -0.0025."""
        x = torch.tensor([[0.0, 0.0]])
        u = torch.zeros(1)
        dx = mountain_car_dynamics(x, u)
        assert torch.isclose(dx[0, 1], torch.tensor(-0.0025))


class TestPendubotDynamics:
    def test_shape(self) -> None:
        x = torch.randn(5, 4)
        u = torch.randn(5)
        dx = pendubot_dynamics(x, u)
        assert dx.shape == (5, 4)

    def test_velocity_passthrough(self) -> None:
        x = torch.tensor([[0.1, 0.2, 0.7, -0.4]])
        u = torch.zeros(1)
        dx = pendubot_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 2])
        assert torch.isclose(dx[0, 1], x[0, 3])


class TestFurutaDynamics:
    def test_shape(self) -> None:
        x = torch.randn(7, 4)
        u = torch.randn(7)
        dx = furuta_dynamics(x, u)
        assert dx.shape == (7, 4)

    def test_velocity_passthrough(self) -> None:
        x = torch.tensor([[0.0, 0.5, 1.0, -0.3]])
        u = torch.zeros(1)
        dx = furuta_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 2])
        assert torch.isclose(dx[0, 1], x[0, 3])

    def test_pendulum_gravity(self) -> None:
        """Displaced pendulum should accelerate due to gravity."""
        x = torch.tensor([[0.0, 0.5, 0.0, 0.0]])
        u = torch.zeros(1)
        dx = furuta_dynamics(x, u)
        assert dx[0, 3].abs() > 0.1


class TestBallBeamDynamics:
    def test_shape(self) -> None:
        x = torch.randn(4, 4)
        u = torch.randn(4)
        dx = ball_beam_dynamics(x, u)
        assert dx.shape == (4, 4)

    def test_velocity_passthrough(self) -> None:
        x = torch.tensor([[0.5, 1.2, 0.1, -0.3]])
        u = torch.zeros(1)
        dx = ball_beam_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 1])
        assert torch.isclose(dx[0, 2], x[0, 3])

    def test_gravity_on_tilted_beam(self) -> None:
        """Tilted beam should cause ball acceleration."""
        x = torch.tensor([[0.0, 0.0, 0.3, 0.0]])
        u = torch.zeros(1)
        dx = ball_beam_dynamics(x, u)
        assert dx[0, 1].abs() > 0.1


class TestQuadrotor2DDynamics:
    def test_shape(self) -> None:
        x = torch.randn(5, 6)
        u = torch.randn(5, 2).abs()
        dx = quadrotor_2d_dynamics(x, u)
        assert dx.shape == (5, 6)

    def test_hover(self) -> None:
        """Equal thrust = mg/2 per rotor at phi=0 should give zero accel."""
        m, g = 1.0, 9.81
        x = torch.zeros(1, 6)
        u = torch.tensor([[m * g / 2, m * g / 2]])
        dx = quadrotor_2d_dynamics(x, u, m=m, g=g)
        assert torch.allclose(dx[0, 2], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(dx[0, 3], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(dx[0, 5], torch.tensor(0.0), atol=1e-5)

    def test_velocity_passthrough(self) -> None:
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.5, -0.1]])
        u = torch.tensor([[0.0, 0.0]])
        dx = quadrotor_2d_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 2])
        assert torch.isclose(dx[0, 1], x[0, 3])
        assert torch.isclose(dx[0, 4], x[0, 5])


class TestTwoLinkArmDynamics:
    def test_shape(self) -> None:
        x = torch.randn(6, 4)
        u = torch.randn(6, 2)
        dx = two_link_arm_dynamics(x, u)
        assert dx.shape == (6, 4)

    def test_velocity_passthrough(self) -> None:
        x = torch.tensor([[0.1, -0.2, 0.5, 0.8]])
        u = torch.zeros(1, 2)
        dx = two_link_arm_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 2])
        assert torch.isclose(dx[0, 1], x[0, 3])


class TestCSTRDynamics:
    def test_shape(self) -> None:
        x = torch.tensor([[5.0, 350.0], [8.0, 320.0], [2.0, 400.0]])
        u = torch.tensor([0.0, 1000.0, -1000.0])
        dx = cstr_dynamics(x, u)
        assert dx.shape == (3, 2)

    def test_feed_flow_at_steady(self) -> None:
        """At feed conditions with no reaction, concentration should not change."""
        # Very low temperature -> negligible reaction rate
        x = torch.tensor([[10.0, 1.0]])  # T=1K -> exp(-8750) ~ 0
        u = torch.zeros(1)
        dx = cstr_dynamics(x, u)
        # dC_A ~ (F/V)*(C_Af - C_A) - ~0 = 0 since C_A = C_Af = 10
        assert torch.allclose(dx[0, 0], torch.tensor(0.0), atol=1e-3)


class TestVanDerPolDynamics:
    def test_shape(self) -> None:
        x = torch.randn(5, 2)
        u = torch.randn(5)
        dx = van_der_pol_dynamics(x, u)
        assert dx.shape == (5, 2)

    def test_equilibrium(self) -> None:
        x = torch.zeros(1, 2)
        u = torch.zeros(1)
        dx = van_der_pol_dynamics(x, u)
        assert torch.allclose(dx, torch.zeros(1, 2), atol=1e-6)

    def test_first_component_is_velocity(self) -> None:
        x = torch.tensor([[1.5, -0.3]])
        u = torch.zeros(1)
        dx = van_der_pol_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 1])


class TestDuffingDynamics:
    def test_shape(self) -> None:
        x = torch.randn(3, 2)
        u = torch.randn(3)
        dx = duffing_dynamics(x, u)
        assert dx.shape == (3, 2)

    def test_equilibrium(self) -> None:
        x = torch.zeros(1, 2)
        u = torch.zeros(1)
        dx = duffing_dynamics(x, u)
        assert torch.allclose(dx, torch.zeros(1, 2), atol=1e-6)

    def test_first_component_is_velocity(self) -> None:
        x = torch.tensor([[0.5, 1.0]])
        u = torch.zeros(1)
        dx = duffing_dynamics(x, u)
        assert torch.isclose(dx[0, 0], x[0, 1])

    def test_restoring_force(self) -> None:
        """Displaced system with no damping/control should have restoring accel."""
        x = torch.tensor([[1.0, 0.0]])
        u = torch.zeros(1)
        dx = duffing_dynamics(x, u, delta=0.0)
        # ddx = -alpha*x - beta*x^3 = -1 - 1 = -2
        assert torch.isclose(dx[0, 1], torch.tensor(-2.0))


class TestLotkaVolterraDynamics:
    def test_shape(self) -> None:
        x = torch.rand(4, 2) + 0.1
        u = torch.rand(4)
        dx = lotka_volterra_dynamics(x, u)
        assert dx.shape == (4, 2)

    def test_zero_populations(self) -> None:
        """Zero populations should give zero derivatives."""
        x = torch.zeros(1, 2)
        u = torch.zeros(1)
        dx = lotka_volterra_dynamics(x, u)
        assert torch.allclose(dx, torch.zeros(1, 2), atol=1e-6)

    def test_harvesting_reduces_prey_growth(self) -> None:
        """Positive harvesting should reduce prey growth rate."""
        x = torch.tensor([[2.0, 0.5]])
        u_no = torch.zeros(1)
        u_yes = torch.tensor([1.0])
        dx_no = lotka_volterra_dynamics(x, u_no)
        dx_yes = lotka_volterra_dynamics(x, u_yes)
        assert dx_yes[0, 0] < dx_no[0, 0]
