"""Tests for environment classes."""

import torch
import pytest

from ncp.envs._base import BaseEnv, StateSpec, ControlSpec
from ncp.envs.pendulum import PendulumEnv
from ncp.envs.unicycle import UnicycleEnv
from ncp.envs.bicycle import BicycleEnv


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
