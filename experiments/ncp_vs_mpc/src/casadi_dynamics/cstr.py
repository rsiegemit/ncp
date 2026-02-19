"""CasADi symbolic dynamics for CSTR."""
import casadi as ca


def cstr_casadi(
    F=1.0, V=1.0, C_Af=10.0, T_f=300.0,
    k0=7.2e10, E_over_R=8750.0, dH=-5e4,
    rho=1000.0, Cp=0.239,
):
    """Return (x, u, xdot) CasADi MX symbolics for CSTR.

    State: [C_A, T], Control: [Q (heat input)]
    """
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)

    C_A = x[0]
    T = x[1]
    Q = u[0]

    rate = k0 * ca.exp(-E_over_R / T) * C_A

    xdot = ca.vertcat(
        (F / V) * (C_Af - C_A) - rate,
        (F / V) * (T_f - T) + (-dH / (rho * Cp)) * rate + Q / (rho * Cp * V),
    )
    return x, u, xdot
