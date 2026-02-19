"""CasADi symbolic dynamics for all 15 systems.

Each module provides a function that returns (x_sym, u_sym, xdot_sym)
as CasADi MX symbolics matching the PyTorch dynamics exactly.
"""

from .pendulum import pendulum_casadi
from .van_der_pol import van_der_pol_casadi
from .duffing import duffing_casadi
from .mountain_car import mountain_car_casadi
from .cstr import cstr_casadi
from .lotka_volterra import lotka_volterra_casadi
from .unicycle import unicycle_casadi
from .bicycle import bicycle_casadi
from .cartpole import cartpole_casadi
from .acrobot import acrobot_casadi
from .pendubot import pendubot_casadi
from .furuta import furuta_casadi
from .ball_beam import ball_beam_casadi
from .two_link_arm import two_link_arm_casadi
from .quadrotor_2d import quadrotor_2d_casadi

CASADI_DYNAMICS = {
    "pendulum": pendulum_casadi,
    "van_der_pol": van_der_pol_casadi,
    "duffing": duffing_casadi,
    "mountain_car": mountain_car_casadi,
    "cstr": cstr_casadi,
    "lotka_volterra": lotka_volterra_casadi,
    "unicycle": unicycle_casadi,
    "bicycle": bicycle_casadi,
    "cartpole": cartpole_casadi,
    "acrobot": acrobot_casadi,
    "pendubot": pendubot_casadi,
    "furuta": furuta_casadi,
    "ball_beam": ball_beam_casadi,
    "two_link_arm": two_link_arm_casadi,
    "quadrotor_2d": quadrotor_2d_casadi,
}
