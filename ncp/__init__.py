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
from ncp.envs._base import BaseEnv, ControlSpec, StateSpec
from ncp.envs.bicycle import BicycleEnv
from ncp.envs.pendulum import PendulumEnv
from ncp.envs.unicycle import UnicycleEnv
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
    # Environments
    "BaseEnv",
    "StateSpec",
    "ControlSpec",
    "PendulumEnv",
    "UnicycleEnv",
    "BicycleEnv",
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
