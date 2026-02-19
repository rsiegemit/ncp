"""CasADi symbolic dynamics for Lotka-Volterra."""
import casadi as ca


def lotka_volterra_casadi(a=1.5, b=1.0, c=3.0, d=1.0):
    """Return (x, u, xdot) CasADi MX symbolics for Lotka-Volterra.

    State: [prey, predator], Control: [harvest_rate]
    """
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)

    prey = x[0]
    predator = x[1]
    harvest = u[0]

    xdot = ca.vertcat(
        a * prey - b * prey * predator - harvest * prey,
        d * prey * predator - c * predator,
    )
    return x, u, xdot
