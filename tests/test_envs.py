"""Tests for environment classes."""

import torch
import pytest

from ncp.envs._base import BaseEnv, StateSpec, ControlSpec
from ncp.envs.pendulum import PendulumEnv
from ncp.envs.unicycle import UnicycleEnv
from ncp.envs.bicycle import BicycleEnv
from ncp.envs.cartpole import CartPoleEnv
from ncp.envs.acrobot import AcrobotEnv
from ncp.envs.mountain_car import MountainCarEnv
from ncp.envs.pendubot import PendubotEnv
from ncp.envs.furuta import FurutaEnv
from ncp.envs.ball_beam import BallBeamEnv
from ncp.envs.quadrotor_2d import Quadrotor2DEnv
from ncp.envs.two_link_arm import TwoLinkArmEnv
from ncp.envs.cstr import CSTREnv
from ncp.envs.van_der_pol import VanDerPolEnv
from ncp.envs.duffing import DuffingEnv
from ncp.envs.lotka_volterra import LotkaVolterraEnv


class TestPendulumEnv:
    def test_creation(self) -> None:
        env = PendulumEnv(num_envs=5)
        assert env.states.shape == (5, 2)
        assert env.state_spec.dim == 2
        assert env.state_spec.wrap_dims == (0,)
        assert env.control_spec.dim == 1

    def test_reset(self) -> None:
        env = PendulumEnv(num_envs=3)
        seed = torch.tensor([[1.0, 0.5], [0.0, 0.0], [-1.0, -0.5]])
        env.reset(seed)
        assert torch.equal(env.states, seed)

    def test_trajectories_shape(self) -> None:
        env = PendulumEnv(num_envs=4, dt=0.1)
        env.reset(torch.randn(4, 2))
        controls = torch.randn(4, 10)
        traj = env.trajectories(controls)
        assert traj.shape == (4, 11, 2)

    def test_sample_trajectory(self) -> None:
        env = PendulumEnv(num_envs=3, dt=0.1)
        env.reset(torch.randn(3, 2))
        actions, traj, dist, taus = env.sample_trajectory(
            time_steps=5, num_samples=10
        )
        assert actions.shape == (3, 5)
        assert traj.shape == (3, 6, 2)
        assert dist.shape == (3,)
        assert taus.shape == (3,)


class TestUnicycleEnv:
    def test_creation(self) -> None:
        env = UnicycleEnv(num_envs=5)
        assert env.states.shape == (5, 3)
        assert env.state_spec.dim == 3
        assert env.state_spec.wrap_dims == (2,)
        assert env.control_spec.dim == 2

    def test_control_bounds(self) -> None:
        env = UnicycleEnv(num_envs=2, max_speed=1.0)
        # v >= 0
        assert env.control_spec.lower_bounds[0].item() == 0.0
        assert env.control_spec.upper_bounds[0].item() == 1.0

    def test_sample_trajectory_shape(self) -> None:
        env = UnicycleEnv(num_envs=2, dt=0.05)
        env.reset(torch.randn(2, 3))
        actions, traj, dist, taus = env.sample_trajectory(
            time_steps=10, num_samples=5
        )
        assert actions.shape == (2, 10, 2)
        assert traj.shape == (2, 11, 3)


class TestBicycleEnv:
    def test_creation(self) -> None:
        env = BicycleEnv(num_envs=3)
        assert env.states.shape == (3, 3)
        assert env.state_spec.wrap_dims == (2,)
        assert env.control_spec.dim == 2


class TestCartPoleEnv:
    def test_creation(self) -> None:
        env = CartPoleEnv(num_envs=5)
        assert env.states.shape == (5, 4)
        assert env.state_spec.dim == 4
        assert env.state_spec.wrap_dims == ()
        assert env.control_spec.dim == 1

    def test_control_bounds(self) -> None:
        env = CartPoleEnv(num_envs=2, max_force=10.0)
        assert env.control_spec.lower_bounds[0].item() == -10.0
        assert env.control_spec.upper_bounds[0].item() == 10.0

    def test_trajectories_shape(self) -> None:
        env = CartPoleEnv(num_envs=3, dt=0.02)
        env.reset(torch.randn(3, 4))
        controls = torch.randn(3, 8)
        traj = env.trajectories(controls)
        assert traj.shape == (3, 9, 4)


class TestAcrobotEnv:
    def test_creation(self) -> None:
        env = AcrobotEnv(num_envs=4)
        assert env.states.shape == (4, 4)
        assert env.state_spec.dim == 4
        assert env.state_spec.wrap_dims == (0, 1)
        assert env.control_spec.dim == 1


class TestMountainCarEnv:
    def test_creation(self) -> None:
        env = MountainCarEnv(num_envs=3)
        assert env.states.shape == (3, 2)
        assert env.state_spec.dim == 2
        assert env.state_spec.wrap_dims == ()
        assert env.control_spec.dim == 1


class TestPendubotEnv:
    def test_creation(self) -> None:
        env = PendubotEnv(num_envs=4)
        assert env.states.shape == (4, 4)
        assert env.state_spec.wrap_dims == (0, 1)
        assert env.control_spec.dim == 1


class TestFurutaEnv:
    def test_creation(self) -> None:
        env = FurutaEnv(num_envs=3)
        assert env.states.shape == (3, 4)
        assert env.state_spec.wrap_dims == (0, 1)
        assert env.control_spec.dim == 1


class TestBallBeamEnv:
    def test_creation(self) -> None:
        env = BallBeamEnv(num_envs=5)
        assert env.states.shape == (5, 4)
        assert env.state_spec.wrap_dims == ()
        assert env.control_spec.dim == 1


class TestQuadrotor2DEnv:
    def test_creation(self) -> None:
        env = Quadrotor2DEnv(num_envs=3)
        assert env.states.shape == (3, 6)
        assert env.state_spec.dim == 6
        assert env.state_spec.wrap_dims == (4,)
        assert env.control_spec.dim == 2

    def test_thrust_bounds(self) -> None:
        env = Quadrotor2DEnv(num_envs=2, mass=1.0, g=9.81)
        assert env.control_spec.lower_bounds[0].item() == 0.0
        assert abs(env.control_spec.upper_bounds[0].item() - 9.81) < 1e-5


class TestTwoLinkArmEnv:
    def test_creation(self) -> None:
        env = TwoLinkArmEnv(num_envs=4)
        assert env.states.shape == (4, 4)
        assert env.state_spec.wrap_dims == (0, 1)
        assert env.control_spec.dim == 2


class TestCSTREnv:
    def test_creation(self) -> None:
        env = CSTREnv(num_envs=3)
        assert env.states.shape == (3, 2)
        assert env.state_spec.wrap_dims == ()
        assert env.control_spec.dim == 1


class TestVanDerPolEnv:
    def test_creation(self) -> None:
        env = VanDerPolEnv(num_envs=4)
        assert env.states.shape == (4, 2)
        assert env.state_spec.wrap_dims == ()
        assert env.control_spec.dim == 1


class TestDuffingEnv:
    def test_creation(self) -> None:
        env = DuffingEnv(num_envs=3)
        assert env.states.shape == (3, 2)
        assert env.state_spec.wrap_dims == ()
        assert env.control_spec.dim == 1


class TestLotkaVolterraEnv:
    def test_creation(self) -> None:
        env = LotkaVolterraEnv(num_envs=5)
        assert env.states.shape == (5, 2)
        assert env.state_spec.wrap_dims == ()
        assert env.control_spec.dim == 1

    def test_non_negative_control(self) -> None:
        env = LotkaVolterraEnv(num_envs=2)
        assert env.control_spec.lower_bounds[0].item() == 0.0


class TestCustomEnv:
    """Test that BaseEnv works with arbitrary dimensionality and no wrapping."""

    def test_no_wrap_dims(self) -> None:
        def linear_dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return -x + u.squeeze(-1).unsqueeze(-1).expand_as(x)

        env = BaseEnv(
            num_envs=4,
            state_spec=StateSpec(dim=5, wrap_dims=()),
            control_spec=ControlSpec(
                dim=1,
                lower_bounds=torch.tensor([-1.0]),
                upper_bounds=torch.tensor([1.0]),
            ),
            dynamics_fn=linear_dynamics,
            dt=0.01,
        )
        assert env.states.shape == (4, 5)
        assert env.state_spec.wrap_dims == ()

        env.reset(torch.randn(4, 5))
        controls = torch.randn(4, 10)
        traj = env.trajectories(controls)
        assert traj.shape == (4, 11, 5)

    def test_high_dimensional(self) -> None:
        """Ensure the framework works for 10-D state spaces."""

        def dummy_dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        env = BaseEnv(
            num_envs=2,
            state_spec=StateSpec(dim=10, wrap_dims=()),
            control_spec=ControlSpec(
                dim=3,
                lower_bounds=torch.tensor([-1.0, -1.0, -1.0]),
                upper_bounds=torch.tensor([1.0, 1.0, 1.0]),
            ),
            dynamics_fn=dummy_dynamics,
            dt=0.01,
        )
        assert env.states.shape == (2, 10)
        env.reset(torch.randn(2, 10))
        controls = torch.randn(2, 5, 3)
        traj = env.trajectories(controls)
        assert traj.shape == (2, 6, 10)
