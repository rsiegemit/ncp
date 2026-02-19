"""MPPI-MPC controller: online MPPI at each simulation step.

Implements the standard MPPI algorithm (Williams et al. 2016, 2017) with:
- Quadratic running cost: x^T Q x + u^T R u
- Per-control-channel noise standard deviation (sigma)
- MPPI temperature parameter (lambda)
- Warm-starting from shifted previous solution

Does NOT use BaseEnv.sample_trajectory with r=0 — uses proper quadratic costs.
"""

import time
import torch
import numpy as np
from typing import Any, Callable


class MPPIMPCController:
    """Online MPPI-MPC controller with quadratic cost.

    At each step, samples K control trajectories, rolls out dynamics,
    scores with quadratic cost, and returns the MPPI-weighted first control.
    """

    def __init__(
        self,
        dynamics_fn: Callable[..., torch.Tensor],
        state_dim: int,
        ctrl_dim: int,
        control_lower: torch.Tensor,
        control_upper: torch.Tensor,
        dt: float,
        horizon_steps: int,
        num_samples: int = 1000,
        sigma: list[float] | torch.Tensor | None = None,
        temperature: float = 1.0,
        Q_diag: list[float] | torch.Tensor | None = None,
        R_diag: list[float] | torch.Tensor | None = None,
        dynamics_kwargs: dict[str, Any] | None = None,
        wrap_dims: tuple[int, ...] = (),
    ):
        self.dynamics_fn = dynamics_fn
        self.state_dim = state_dim
        self.ctrl_dim = ctrl_dim
        self.dt = dt
        self.N = horizon_steps
        self.K = num_samples
        self.temperature = temperature
        self.dynamics_kwargs = dynamics_kwargs or {}
        self.wrap_dims = wrap_dims

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Control bounds
        self._lb = control_lower.to(self.device)
        self._ub = control_upper.to(self.device)

        # Per-channel noise standard deviation
        if sigma is None:
            # Default: 20% of control range per channel
            self._sigma = ((self._ub - self._lb) * 0.2).to(self.device)
        elif isinstance(sigma, list):
            self._sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        else:
            self._sigma = sigma.to(self.device)

        # Cost weight matrices (stored as diagonal vectors for efficiency)
        if Q_diag is None:
            self._Q = torch.ones(state_dim, device=self.device)
        elif isinstance(Q_diag, list):
            self._Q = torch.tensor(Q_diag, dtype=torch.float32, device=self.device)
        else:
            self._Q = Q_diag.to(self.device)

        if R_diag is None:
            self._R = torch.ones(ctrl_dim, device=self.device) * 0.01
        elif isinstance(R_diag, list):
            self._R = torch.tensor(R_diag, dtype=torch.float32, device=self.device)
        else:
            self._R = R_diag.to(self.device)

        # Mean control trajectory (warm-start buffer)
        if ctrl_dim == 1:
            self._U_mean = torch.zeros(horizon_steps, device=self.device)
        else:
            self._U_mean = torch.zeros(horizon_steps, ctrl_dim, device=self.device)

    def reset(self):
        """Reset warm-start buffer."""
        if self.ctrl_dim == 1:
            self._U_mean = torch.zeros(self.N, device=self.device)
        else:
            self._U_mean = torch.zeros(self.N, self.ctrl_dim, device=self.device)

    @torch.no_grad()
    def step(self, state: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute one control action via MPPI.

        Args:
            state: (state_dim,) current state tensor

        Returns:
            (u, info) where u is (ctrl_dim,) or scalar control,
            info contains solve_time and planned trajectory.
        """
        device = self.device
        K = self.K
        N = self.N
        nx = self.state_dim
        nu = self.ctrl_dim
        x0 = state.to(device)

        t0 = time.time()

        # Sample noise: (K, N, nu) or (K, N) for 1D
        if nu == 1:
            noise = torch.randn(K, N, device=device) * self._sigma[0]
            # Perturbed controls around mean
            U_samples = self._U_mean.unsqueeze(0) + noise  # (K, N)
            U_samples = U_samples.clamp(self._lb.item(), self._ub.item())
        else:
            noise = torch.randn(K, N, nu, device=device) * self._sigma  # broadcast sigma
            U_samples = self._U_mean.unsqueeze(0) + noise  # (K, N, nu)
            U_samples = torch.max(torch.min(U_samples, self._ub), self._lb)

        # Rollout dynamics for all K samples in parallel
        # x: (K, nx)
        x = x0.unsqueeze(0).expand(K, -1).clone()
        total_cost = torch.zeros(K, device=device)

        # Also track best trajectory for info
        traj = torch.empty(K, N + 1, nx, device=device)
        traj[:, 0] = x

        for t in range(N):
            if nu == 1:
                u_t = U_samples[:, t]  # (K,)
            else:
                u_t = U_samples[:, t]  # (K, nu)

            # Running cost: x^T Q x + u^T R u
            state_cost = (x ** 2 * self._Q).sum(dim=-1)  # (K,)
            if nu == 1:
                ctrl_cost = u_t ** 2 * self._R[0]  # (K,)
            else:
                ctrl_cost = (u_t ** 2 * self._R).sum(dim=-1)  # (K,)
            total_cost += state_cost + ctrl_cost

            # Dynamics step: x_next = x + f(x, u) * dt
            dx = self.dynamics_fn(x, u_t, **self.dynamics_kwargs)
            x = x + dx * self.dt

            # Wrap angles if needed
            if self.wrap_dims:
                for d in self.wrap_dims:
                    x[:, d] = ((x[:, d] + torch.pi) % (2 * torch.pi)) - torch.pi

            traj[:, t + 1] = x

        # Terminal cost: 10 * x^T Q x
        terminal_cost = 10.0 * (x ** 2 * self._Q).sum(dim=-1)
        total_cost += terminal_cost

        # MPPI weighting: w_k = exp(-1/lambda * S_k), normalized
        # Shift costs for numerical stability
        min_cost = total_cost.min()
        weights = torch.exp(-(total_cost - min_cost) / self.temperature)
        weights = weights / weights.sum()

        # Weighted mean control
        if nu == 1:
            U_new = (weights.unsqueeze(1) * U_samples).sum(dim=0)  # (N,)
        else:
            U_new = (weights.unsqueeze(1).unsqueeze(2) * U_samples).sum(dim=0)  # (N, nu)

        solve_time = time.time() - t0

        # Extract first control
        if nu == 1:
            u_out = U_new[0].cpu()
        else:
            u_out = U_new[0].cpu()

        # Get best trajectory for info
        best_idx = total_cost.argmin()
        best_traj = traj[best_idx].cpu()

        # Warm-start: shift mean forward, pad last step with zeros
        if nu == 1:
            self._U_mean = torch.cat([U_new[1:], torch.zeros(1, device=device)])
        else:
            self._U_mean = torch.cat([U_new[1:], torch.zeros(1, nu, device=device)])

        info = {
            "solve_time": solve_time,
            "planned_trajectory": best_traj,
            "best_cost": total_cost[best_idx].item(),
            "mean_cost": total_cost.mean().item(),
        }
        return u_out, info
