"""CasADi symbolic dynamics for Mountain Car."""
import casadi as ca


def mountain_car_casadi():
    """Return (x, u, xdot) CasADi MX symbolics for Mountain Car.

    State: [position, velocity], Control: [force]
    """
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)

    velocity = x[1]
    force = u[0]

    xdot = ca.vertcat(
        velocity,
        force - 0.0025 * ca.cos(3.0 * x[0]),
    )
    return x, u, xdot
