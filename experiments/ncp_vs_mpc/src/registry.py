"""System registry: maps system names to env classes, dynamics functions, and specs."""

import torch
from ncp import (
    StateSpec,
    ControlSpec,
    # Env classes
    PendulumEnv,
    VanDerPolEnv,
    DuffingEnv,
    MountainCarEnv,
    CSTREnv,
    LotkaVolterraEnv,
    UnicycleEnv,
    BicycleEnv,
    CartPoleEnv,
    AcrobotEnv,
    PendubotEnv,
    FurutaEnv,
    BallBeamEnv,
    TwoLinkArmEnv,
    Quadrotor2DEnv,
    # Dynamics functions
    inverted_pendulum_2d_torch,
    van_der_pol_dynamics,
    duffing_dynamics,
    mountain_car_dynamics,
    cstr_dynamics,
    lotka_volterra_dynamics,
    unicycle_derivatives,
    bicycle_derivatives,
    cartpole_dynamics,
    acrobot_dynamics,
    pendubot_dynamics,
    furuta_dynamics,
    ball_beam_dynamics,
    two_link_arm_dynamics,
    quadrotor_2d_dynamics,
)
from dataclasses import dataclass
from typing import Any, Callable, Type
from ncp.envs._base import BaseEnv


@dataclass(frozen=True)
class SystemInfo:
    name: str
    env_class: Type[BaseEnv]
    dynamics_fn: Callable[..., torch.Tensor]
    state_spec: StateSpec
    control_spec: ControlSpec
    default_dt: float
    dynamics_kwargs: dict[str, Any] | None = None


SYSTEMS: dict[str, SystemInfo] = {
    "pendulum": SystemInfo(
        name="pendulum",
        env_class=PendulumEnv,
        dynamics_fn=inverted_pendulum_2d_torch,
        state_spec=StateSpec(dim=2, wrap_dims=(0,)),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-2.0]), upper_bounds=torch.tensor([2.0])),
        default_dt=0.1,
    ),
    "van_der_pol": SystemInfo(
        name="van_der_pol",
        env_class=VanDerPolEnv,
        dynamics_fn=van_der_pol_dynamics,
        state_spec=StateSpec(dim=2, wrap_dims=()),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-5.0]), upper_bounds=torch.tensor([5.0])),
        default_dt=0.05,
    ),
    "duffing": SystemInfo(
        name="duffing",
        env_class=DuffingEnv,
        dynamics_fn=duffing_dynamics,
        state_spec=StateSpec(dim=2, wrap_dims=()),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-10.0]), upper_bounds=torch.tensor([10.0])),
        default_dt=0.05,
    ),
    "mountain_car": SystemInfo(
        name="mountain_car",
        env_class=MountainCarEnv,
        dynamics_fn=mountain_car_dynamics,
        state_spec=StateSpec(dim=2, wrap_dims=()),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-1.0]), upper_bounds=torch.tensor([1.0])),
        default_dt=0.1,
    ),
    "cstr": SystemInfo(
        name="cstr",
        env_class=CSTREnv,
        dynamics_fn=cstr_dynamics,
        state_spec=StateSpec(dim=2, wrap_dims=()),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-5000.0]), upper_bounds=torch.tensor([5000.0])),
        default_dt=0.01,
    ),
    "lotka_volterra": SystemInfo(
        name="lotka_volterra",
        env_class=LotkaVolterraEnv,
        dynamics_fn=lotka_volterra_dynamics,
        state_spec=StateSpec(dim=2, wrap_dims=()),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([0.0]), upper_bounds=torch.tensor([3.0])),
        default_dt=0.05,
    ),
    "unicycle": SystemInfo(
        name="unicycle",
        env_class=UnicycleEnv,
        dynamics_fn=unicycle_derivatives,
        state_spec=StateSpec(dim=3, wrap_dims=(2,)),
        control_spec=ControlSpec(dim=2, lower_bounds=torch.tensor([0.0, -1.0]), upper_bounds=torch.tensor([1.0, 1.0])),
        default_dt=0.1,
    ),
    "bicycle": SystemInfo(
        name="bicycle",
        env_class=BicycleEnv,
        dynamics_fn=bicycle_derivatives,
        state_spec=StateSpec(dim=3, wrap_dims=(2,)),
        control_spec=ControlSpec(dim=2, lower_bounds=torch.tensor([0.0, -0.4]), upper_bounds=torch.tensor([1.0, 0.4])),
        default_dt=0.1,
    ),
    "cartpole": SystemInfo(
        name="cartpole",
        env_class=CartPoleEnv,
        dynamics_fn=cartpole_dynamics,
        state_spec=StateSpec(dim=4, wrap_dims=()),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-10.0]), upper_bounds=torch.tensor([10.0])),
        default_dt=0.02,
    ),
    "acrobot": SystemInfo(
        name="acrobot",
        env_class=AcrobotEnv,
        dynamics_fn=acrobot_dynamics,
        state_spec=StateSpec(dim=4, wrap_dims=(0, 1)),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-1.0]), upper_bounds=torch.tensor([1.0])),
        default_dt=0.05,
    ),
    "pendubot": SystemInfo(
        name="pendubot",
        env_class=PendubotEnv,
        dynamics_fn=pendubot_dynamics,
        state_spec=StateSpec(dim=4, wrap_dims=(0, 1)),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-5.0]), upper_bounds=torch.tensor([5.0])),
        default_dt=0.05,
    ),
    "furuta": SystemInfo(
        name="furuta",
        env_class=FurutaEnv,
        dynamics_fn=furuta_dynamics,
        state_spec=StateSpec(dim=4, wrap_dims=(0, 1)),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-5.0]), upper_bounds=torch.tensor([5.0])),
        default_dt=0.02,
    ),
    "ball_beam": SystemInfo(
        name="ball_beam",
        env_class=BallBeamEnv,
        dynamics_fn=ball_beam_dynamics,
        state_spec=StateSpec(dim=4, wrap_dims=()),
        control_spec=ControlSpec(dim=1, lower_bounds=torch.tensor([-5.0]), upper_bounds=torch.tensor([5.0])),
        default_dt=0.02,
    ),
    "two_link_arm": SystemInfo(
        name="two_link_arm",
        env_class=TwoLinkArmEnv,
        dynamics_fn=two_link_arm_dynamics,
        state_spec=StateSpec(dim=4, wrap_dims=(0, 1)),
        control_spec=ControlSpec(dim=2, lower_bounds=torch.tensor([-10.0, -10.0]), upper_bounds=torch.tensor([10.0, 10.0])),
        default_dt=0.02,
    ),
    "quadrotor_2d": SystemInfo(
        name="quadrotor_2d",
        env_class=Quadrotor2DEnv,
        dynamics_fn=quadrotor_2d_dynamics,
        state_spec=StateSpec(dim=6, wrap_dims=(4,)),
        control_spec=ControlSpec(dim=2, lower_bounds=torch.tensor([0.0, 0.0]), upper_bounds=torch.tensor([9.81, 9.81])),
        default_dt=0.02,
    ),
}

# Ordered list matching SLURM array indices 0-14
SYSTEM_ORDER = [
    "pendulum", "van_der_pol", "duffing", "mountain_car", "cstr",
    "lotka_volterra", "unicycle", "bicycle", "cartpole", "acrobot",
    "pendubot", "furuta", "ball_beam", "two_link_arm", "quadrotor_2d",
]


def get_system(name: str) -> SystemInfo:
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system: {name}. Available: {list(SYSTEMS.keys())}")
    return SYSTEMS[name]


def get_system_by_index(idx: int) -> SystemInfo:
    if idx < 0 or idx >= len(SYSTEM_ORDER):
        raise ValueError(f"Index {idx} out of range [0, {len(SYSTEM_ORDER) - 1}]")
    return SYSTEMS[SYSTEM_ORDER[idx]]
