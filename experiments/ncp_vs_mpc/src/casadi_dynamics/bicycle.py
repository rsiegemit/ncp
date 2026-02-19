"""CasADi symbolic dynamics for Kinematic Bicycle."""
import casadi as ca


def bicycle_casadi(wheelbase=2.0):
    """Return (x, u, xdot) CasADi MX symbolics for kinematic bicycle.

    State: [x, y, theta], Control: [v, delta]
    """
    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 2)

    theta = x[2]
    v = u[0]
    delta = u[1]

    xdot = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        v / wheelbase * ca.tan(delta),
    )
    return x, u, xdot
