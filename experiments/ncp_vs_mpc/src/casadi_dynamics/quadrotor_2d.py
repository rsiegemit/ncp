"""CasADi symbolic dynamics for 2D Quadrotor (PVTOL)."""
import casadi as ca


def quadrotor_2d_casadi(m=1.0, J=0.01, d=0.2, g=9.81):
    """Return (x, u, xdot) CasADi MX symbolics for 2D Quadrotor.

    State: [x, y, x_dot, y_dot, phi, phi_dot], Control: [f1, f2]
    """
    x = ca.MX.sym("x", 6)
    u = ca.MX.sym("u", 2)

    phi = x[4]
    phi_dot = x[5]
    f1 = u[0]
    f2 = u[1]

    total_thrust = f1 + f2

    xdot = ca.vertcat(
        x[2],                                    # dx
        x[3],                                    # dy
        -total_thrust * ca.sin(phi) / m,         # ddx
        total_thrust * ca.cos(phi) / m - g,      # ddy
        phi_dot,                                  # dphi
        (f1 - f2) * d / J,                       # ddphi
    )
    return x, u, xdot
