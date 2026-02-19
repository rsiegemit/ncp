"""CasADi/IPOPT NMPC controller: multiple-shooting formulation with Euler integration.

Quadratic stage cost ||x||_Q^2 + ||u||_R^2, terminal weight on ||x_T||_Qf^2.
IPOPT with warm-starting from shifted previous solution.
"""

import time
import numpy as np
from typing import Any, Callable

try:
    import casadi as ca
except ImportError:
    raise ImportError("CasADi not installed. Run: pip install --user casadi")


class CasADiNMPCController:
    """CasADi/IPOPT nonlinear MPC controller.

    Uses multiple-shooting with Euler integration matching NCP.
    """

    def __init__(
        self,
        casadi_dynamics_fn: Callable,
        state_dim: int,
        ctrl_dim: int,
        control_lower: np.ndarray,
        control_upper: np.ndarray,
        dt: float,
        horizon_steps: int,
        Q_diag: np.ndarray | None = None,
        R_diag: np.ndarray | None = None,
        Qf_diag: np.ndarray | None = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        dynamics_kwargs: dict[str, Any] | None = None,
    ):
        self.state_dim = state_dim
        self.ctrl_dim = ctrl_dim
        self.dt = dt
        self.N = horizon_steps
        self.max_iter = max_iter

        # Cost weights
        if Q_diag is None:
            Q_diag = np.ones(state_dim)
        if R_diag is None:
            R_diag = np.ones(ctrl_dim) * 0.01
        if Qf_diag is None:
            Qf_diag = Q_diag * 10.0

        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        self.Qf = np.diag(Qf_diag)

        # Build CasADi dynamics
        kwargs = dynamics_kwargs or {}
        x_sym, u_sym, xdot_sym = casadi_dynamics_fn(**kwargs)

        # Euler integration: x_next = x + xdot * dt
        x_next = x_sym + xdot_sym * dt
        self.f_discrete = ca.Function("f_discrete", [x_sym, u_sym], [x_next])

        # Build NLP
        self._build_nlp(control_lower, control_upper, tol)

        # Warm-start buffers
        self._prev_x_sol = None
        self._prev_u_sol = None

        # Stats
        self._last_ipopt_iters = 0
        self._last_converged = True

    def _build_nlp(self, u_lb, u_ub, tol):
        """Build the multiple-shooting NLP."""
        N = self.N
        nx = self.state_dim
        nu = self.ctrl_dim

        # Decision variables
        X = ca.MX.sym("X", nx, N + 1)  # states at each shooting node
        U = ca.MX.sym("U", nu, N)       # controls at each interval
        P = ca.MX.sym("P", nx)           # parameter: current state

        # Flatten for solver
        w = []
        w0 = []
        lbw = []
        ubw = []
        g = []
        lbg = []
        ubg = []

        Q_ca = ca.DM(self.Q)
        R_ca = ca.DM(self.R)
        Qf_ca = ca.DM(self.Qf)

        cost = 0

        for k in range(N):
            # State variables at node k
            xk = ca.MX.sym(f"x_{k}", nx)
            w.append(xk)
            if k == 0:
                lbw += [-ca.inf] * nx
                ubw += [ca.inf] * nx
            else:
                lbw += [-ca.inf] * nx
                ubw += [ca.inf] * nx
            w0 += [0.0] * nx

            # Control variables at interval k
            uk = ca.MX.sym(f"u_{k}", nu)
            w.append(uk)
            lbw += list(u_lb)
            ubw += list(u_ub)
            w0 += [0.0] * nu

            # Stage cost
            cost += ca.mtimes([xk.T, Q_ca, xk]) + ca.mtimes([uk.T, R_ca, uk])

            # Dynamics constraint (shooting): x_{k+1} = f(x_k, u_k)
            x_next = self.f_discrete(xk, uk)

            if k == 0:
                # Initial state constraint
                g.append(xk - P)
                lbg += [0.0] * nx
                ubg += [0.0] * nx

        # Terminal state
        xN = ca.MX.sym(f"x_{N}", nx)
        w.append(xN)
        lbw += [-ca.inf] * nx
        ubw += [ca.inf] * nx
        w0 += [0.0] * nx

        # Terminal cost
        cost += ca.mtimes([xN.T, Qf_ca, xN])

        # Shooting constraints: link x_{k+1} to dynamics(x_k, u_k)
        # We need to reconstruct from the flat w vector
        # Re-build with proper indexing
        w_all = ca.vertcat(*w)

        # Actually, let me rebuild this more carefully with proper NLP structure
        # Reset
        w = []
        w0_list = []
        lbw = []
        ubw = []
        g = []
        lbg = []
        ubg = []
        cost = 0

        # Initial state as first variable
        X0 = ca.MX.sym("X_0", nx)
        w.append(X0)
        lbw += [-ca.inf] * nx
        ubw += [ca.inf] * nx
        w0_list += [0.0] * nx

        # Initial state = parameter constraint
        g.append(X0 - P)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        Xk = X0
        for k in range(N):
            # Control at step k
            Uk = ca.MX.sym(f"U_{k}", nu)
            w.append(Uk)
            lbw += list(u_lb)
            ubw += list(u_ub)
            w0_list += [0.0] * nu

            # Stage cost
            cost += ca.mtimes([Xk.T, Q_ca, Xk]) + ca.mtimes([Uk.T, R_ca, Uk])

            # Next state from dynamics
            Xk_next = self.f_discrete(Xk, Uk)

            # Next state variable (shooting node)
            Xk1 = ca.MX.sym(f"X_{k+1}", nx)
            w.append(Xk1)
            lbw += [-ca.inf] * nx
            ubw += [ca.inf] * nx
            w0_list += [0.0] * nx

            # Continuity constraint
            g.append(Xk1 - Xk_next)
            lbg += [0.0] * nx
            ubg += [0.0] * nx

            Xk = Xk1

        # Terminal cost
        cost += ca.mtimes([Xk.T, Qf_ca, Xk])

        # Build NLP
        nlp = {
            "x": ca.vertcat(*w),
            "f": cost,
            "g": ca.vertcat(*g),
            "p": P,
        }

        opts = {
            "ipopt.max_iter": self.max_iter,
            "ipopt.tol": tol,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_init": 1e-3,
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self.lbw = np.array(lbw)
        self.ubw = np.array(ubw)
        self.lbg = np.array(lbg)
        self.ubg = np.array(ubg)
        self.w0 = np.array(w0_list)

        # Store dimensions for warm-start extraction
        self._nx = nx
        self._nu = nu
        self._n_vars = len(w0_list)

    def reset(self):
        """Reset warm-start buffers."""
        self._prev_x_sol = None
        self._prev_u_sol = None

    def _extract_solution(self, sol_x):
        """Extract state and control trajectories from flat solution vector."""
        nx = self._nx
        nu = self._nu
        N = self.N

        x_sol = []
        u_sol = []
        idx = 0

        # X_0
        x_sol.append(sol_x[idx:idx + nx])
        idx += nx

        for k in range(N):
            # U_k
            u_sol.append(sol_x[idx:idx + nu])
            idx += nu
            # X_{k+1}
            x_sol.append(sol_x[idx:idx + nx])
            idx += nx

        return np.array(x_sol), np.array(u_sol)

    def _build_warm_start(self, current_state):
        """Build warm-start from shifted previous solution."""
        nx = self._nx
        nu = self._nu
        N = self.N

        if self._prev_x_sol is None or self._prev_u_sol is None:
            return self.w0.copy()

        # Shift: x[1:] + repeat last, u[1:] + repeat last
        x_shifted = np.zeros((N + 1, nx))
        u_shifted = np.zeros((N, nu))

        x_shifted[0] = current_state
        for i in range(N - 1):
            x_shifted[i + 1] = self._prev_x_sol[i + 2] if i + 2 < len(self._prev_x_sol) else self._prev_x_sol[-1]
            u_shifted[i] = self._prev_u_sol[i + 1] if i + 1 < len(self._prev_u_sol) else self._prev_u_sol[-1]
        x_shifted[N] = self._prev_x_sol[-1] if len(self._prev_x_sol) > 0 else np.zeros(nx)
        u_shifted[N - 1] = self._prev_u_sol[-1] if len(self._prev_u_sol) > 0 else np.zeros(nu)

        # Pack into flat vector
        w0 = np.zeros(self._n_vars)
        idx = 0
        w0[idx:idx + nx] = x_shifted[0]
        idx += nx
        for k in range(N):
            w0[idx:idx + nu] = u_shifted[k]
            idx += nu
            w0[idx:idx + nx] = x_shifted[k + 1]
            idx += nx

        return w0

    def step(self, state: np.ndarray) -> tuple[np.ndarray, dict]:
        """Compute one control action via NMPC.

        Args:
            state: (state_dim,) current state

        Returns:
            (u, info) where u is (ctrl_dim,) control,
            info contains solve_time, ipopt_iterations, converged.
        """
        w0 = self._build_warm_start(state)

        t0 = time.time()
        sol = self.solver(
            x0=w0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
            p=state,
        )
        solve_time = time.time() - t0

        sol_x = np.array(sol["x"]).flatten()
        x_sol, u_sol = self._extract_solution(sol_x)

        # Store for warm-starting
        self._prev_x_sol = x_sol
        self._prev_u_sol = u_sol

        # Get IPOPT stats
        stats = self.solver.stats()
        ipopt_iters = stats.get("iter_count", 0)
        return_status = stats.get("return_status", "unknown")
        converged = return_status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

        self._last_ipopt_iters = ipopt_iters
        self._last_converged = converged

        info = {
            "solve_time": solve_time,
            "ipopt_iterations": ipopt_iters,
            "converged": converged,
            "return_status": return_status,
            "planned_trajectory": x_sol,
        }
        return u_sol[0], info
