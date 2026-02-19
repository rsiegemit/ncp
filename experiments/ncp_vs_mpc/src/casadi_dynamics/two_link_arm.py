"""CasADi symbolic dynamics for Two-Link Arm."""
import casadi as ca


def two_link_arm_casadi(m1=1.0, m2=1.0, l1=1.0, l2=1.0, lc1=0.5, lc2=0.5, I1=1.0, I2=1.0, g=9.81):
    """Return (x, u, xdot) CasADi MX symbolics for Two-Link Arm.

    State: [q1, q2, q1_dot, q2_dot], Control: [tau1, tau2]
    """
    x = ca.MX.sym("x", 4)
    u = ca.MX.sym("u", 2)

    q1 = x[0]
    q2 = x[1]
    q1_dot = x[2]
    q2_dot = x[3]
    tau1 = u[0]
    tau2 = u[1]

    c2 = ca.cos(q2)
    s2 = ca.sin(q2)
    s1 = ca.sin(q1)
    s12 = ca.sin(q1 + q2)

    h = m2 * l1 * lc2
    d11 = I1 + I2 + m2 * l1**2 + 2.0 * h * c2
    d12 = I2 + h * c2
    d22 = I2

    c1 = -h * s2 * q2_dot * (2.0 * q1_dot + q2_dot)
    c2_term = h * s2 * q1_dot**2

    g1 = (m1 * lc1 + m2 * l1) * g * s1 + m2 * g * lc2 * s12
    g2 = m2 * g * lc2 * s12

    # B = I(2x2)
    rhs1 = tau1 - c1 - g1
    rhs2 = tau2 - c2_term - g2

    det = d11 * d22 - d12 * d12
    q1_ddot = (d22 * rhs1 - d12 * rhs2) / det
    q2_ddot = (-d12 * rhs1 + d11 * rhs2) / det

    xdot = ca.vertcat(q1_dot, q2_dot, q1_ddot, q2_ddot)
    return x, u, xdot
