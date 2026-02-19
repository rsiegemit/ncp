"""CasADi symbolic dynamics for Unicycle."""
import casadi as ca


def unicycle_casadi():
    """Return (x, u, xdot) CasADi MX symbolics for Unicycle (no wind).

    State: [x, y, theta], Control: [v, omega]
    """
    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 2)

    theta = x[2]
    v = u[0]
    omega = u[1]

    xdot = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
    )
    return x, u, xdot
