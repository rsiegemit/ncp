"""CasADi symbolic dynamics for Ball-and-Beam."""
import casadi as ca


def ball_beam_casadi(m=0.05, J=0.02, g=9.81):
    """Return (x, u, xdot) CasADi MX symbolics for Ball-and-Beam.

    State: [r, r_dot, theta, theta_dot], Control: [beam_torque]
    """
    x = ca.MX.sym("x", 4)
    u = ca.MX.sym("u", 1)

    r = x[0]
    r_dot = x[1]
    theta = x[2]
    theta_dot = x[3]
    tau = u[0]

    xdot = ca.vertcat(
        r_dot,
        r * theta_dot**2 - g * ca.sin(theta),
        theta_dot,
        (tau - 2.0 * m * r * r_dot * theta_dot - m * g * r * ca.cos(theta)) / (m * r**2 + J),
    )
    return x, u, xdot
