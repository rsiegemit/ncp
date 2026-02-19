"""CasADi symbolic dynamics for CartPole."""
import casadi as ca


def cartpole_casadi(m_c=1.0, m_p=0.1, l=0.5, g=9.81):
    """Return (x, u, xdot) CasADi MX symbolics for CartPole.

    State: [x, x_dot, theta, theta_dot], Control: [force]
    """
    x = ca.MX.sym("x", 4)
    u = ca.MX.sym("u", 1)

    pos_dot = x[1]
    theta = x[2]
    theta_dot = x[3]
    F = u[0]

    sin_th = ca.sin(theta)
    cos_th = ca.cos(theta)
    total_mass = m_c + m_p

    theta_ddot = (
        g * sin_th
        - cos_th * (F + m_p * l * theta_dot**2 * sin_th) / total_mass
    ) / (l * (4.0 / 3.0 - m_p * cos_th**2 / total_mass))

    x_ddot = (
        F + m_p * l * (theta_dot**2 * sin_th - theta_ddot * cos_th)
    ) / total_mass

    xdot = ca.vertcat(pos_dot, x_ddot, theta_dot, theta_ddot)
    return x, u, xdot
