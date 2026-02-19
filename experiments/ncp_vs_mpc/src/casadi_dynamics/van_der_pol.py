"""CasADi symbolic dynamics for Van der Pol oscillator."""
import casadi as ca


def van_der_pol_casadi(mu=1.0):
    """Return (x, u, xdot) CasADi MX symbolics for Van der Pol.

    State: [x, x_dot], Control: [force]
    """
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)

    x0 = x[0]
    x1 = x[1]
    force = u[0]

    xdot = ca.vertcat(
        x1,
        mu * (1.0 - x0**2) * x1 - x0 + force,
    )
    return x, u, xdot
