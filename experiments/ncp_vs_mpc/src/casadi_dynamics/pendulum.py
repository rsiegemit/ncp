"""CasADi symbolic dynamics for inverted pendulum."""
import casadi as ca


def pendulum_casadi(m=0.1, l=10.0, g=9.81):
    """Return (x, u, xdot) CasADi MX symbolics for the inverted pendulum.

    State: [theta, theta_dot], Control: [torque]
    """
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)

    theta = x[0]
    theta_dot = x[1]
    torque = u[0]

    xdot = ca.vertcat(
        theta_dot,
        (g / l) * ca.sin(theta) + (torque / (m * l)) * ca.fabs(ca.cos(theta)),
    )
    return x, u, xdot
