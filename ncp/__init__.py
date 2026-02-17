"""NCP — Nonparametric Chain Policies for practical stabilization."""

from ncp.algorithm.builder import NCPBuilder
from ncp.algorithm.result import NCPResult
from ncp.algorithm.search import certify_alpha, search_alpha_parallel
from ncp.algorithm.mppi import find_path, find_path_reuse
from ncp.dynamics.bicycle import bicycle_derivatives
from ncp.dynamics.pendulum import (
    inverted_pendulum_2d_torch,
    simplified_pendulum_derivatives,
)
from ncp.dynamics.unicycle import unicycle_derivatives, unicycle_derivatives_radial
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
from ncp.envs._base import BaseEnv, ControlSpec, StateSpec
from ncp.envs.bicycle import BicycleEnv
from ncp.envs.pendulum import PendulumEnv
from ncp.envs.unicycle import UnicycleEnv
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
from ncp.evaluation.controller import NCPController
from ncp.evaluation.simulate import simulate
from ncp.geometry.grid import generate_initial_grid, subdivide_cells
from ncp.geometry.hypercubes import find_hypercubes
from ncp.geometry.intersections import (
    count_hypercube_intersections,
    find_hypercube_intersections,
)
from ncp.utils.wrapping import wrap_angles
from ncp.visualization.phase_portrait import plot_cells
from ncp.visualization.trajectory import plot_trajectories

__all__ = [
    # Builder & result
    "NCPBuilder",
    "NCPResult",
    # Algorithm
    "search_alpha_parallel",
    "certify_alpha",
    "find_path",
    "find_path_reuse",
    # Dynamics
    "inverted_pendulum_2d_torch",
    "simplified_pendulum_derivatives",
    "unicycle_derivatives",
    "unicycle_derivatives_radial",
    "bicycle_derivatives",
    "cartpole_dynamics",
    "acrobot_dynamics",
    "mountain_car_dynamics",
    "pendubot_dynamics",
    "furuta_dynamics",
    "ball_beam_dynamics",
    "quadrotor_2d_dynamics",
    "two_link_arm_dynamics",
    "cstr_dynamics",
    "van_der_pol_dynamics",
    "duffing_dynamics",
    "lotka_volterra_dynamics",
    # Environments
    "BaseEnv",
    "StateSpec",
    "ControlSpec",
    "PendulumEnv",
    "UnicycleEnv",
    "BicycleEnv",
    "CartPoleEnv",
    "AcrobotEnv",
    "MountainCarEnv",
    "PendubotEnv",
    "FurutaEnv",
    "BallBeamEnv",
    "Quadrotor2DEnv",
    "TwoLinkArmEnv",
    "CSTREnv",
    "VanDerPolEnv",
    "DuffingEnv",
    "LotkaVolterraEnv",
    # Evaluation
    "NCPController",
    "simulate",
    # Geometry
    "generate_initial_grid",
    "subdivide_cells",
    "find_hypercubes",
    "count_hypercube_intersections",
    "find_hypercube_intersections",
    # Utilities
    "wrap_angles",
    # Visualization
    "plot_cells",
    "plot_trajectories",
]
