"""CasADi symbolic dynamics for Duffing oscillator."""
import casadi as ca


def duffing_casadi(alpha=1.0, beta=1.0, delta=0.3):
    """Return (x, u, xdot) CasADi MX symbolics for Duffing oscillator.

    State: [x, x_dot], Control: [force]
    """
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)

    x0 = x[0]
    x1 = x[1]
    force = u[0]

    xdot = ca.vertcat(
        x1,
        -delta * x1 - alpha * x0 - beta * x0**3 + force,
    )
    return x, u, xdot
