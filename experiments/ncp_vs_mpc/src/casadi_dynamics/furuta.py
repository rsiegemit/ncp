"""CasADi symbolic dynamics for Furuta pendulum."""
import casadi as ca


def furuta_casadi(m_arm=0.095, m_pend=0.024, L_arm=0.085, l_pend=0.129,
                  J_arm=5.72e-5, J_pend=3.33e-5, g=9.81):
    """Return (x, u, xdot) CasADi MX symbolics for Furuta pendulum.

    State: [theta_arm, theta_pend, dtheta_arm, dtheta_pend], Control: [motor_torque]
    """
    x = ca.MX.sym("x", 4)
    u = ca.MX.sym("u", 1)

    th_a_dot = x[2]
    th_p = x[1]
    th_p_dot = x[3]
    tau = u[0]

    cp = ca.cos(th_p)
    sp = ca.sin(th_p)

    a = J_arm + m_pend * L_arm**2
    b = m_pend * L_arm * l_pend
    d = J_pend + m_pend * l_pend**2

    det = a * d - (b * cp)**2

    # Coriolis and gravity
    c1 = -2.0 * b * sp * cp * th_a_dot * th_p_dot - b * sp * th_p_dot**2
    c2 = b * sp * cp * th_a_dot**2
    g2 = m_pend * g * l_pend * sp

    rhs1 = tau + c1
    rhs2 = -c2 - g2

    th_a_ddot = (d * rhs1 - b * cp * rhs2) / det
    th_p_ddot = (-b * cp * rhs1 + a * rhs2) / det

    xdot = ca.vertcat(th_a_dot, th_p_dot, th_a_ddot, th_p_ddot)
    return x, u, xdot
