#!/usr/bin/env python3
"""CLI: Evaluate all 3 controllers on shared ICs with uniform step() interface.

Two NCP evaluation modes:
1. Native: Apply full stored control sequence (up to tau steps), then re-query.
2. Step-by-step: Apply one control, re-query every step. Apples-to-apples with MPC.
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_system, get_system_by_index
from src.config import load_config
from src.initial_conditions import load_ics
from src.metrics import ClosedLoopMetrics, compute_closed_loop_metrics
from src.mppi_mpc import MPPIMPCController
from src.casadi_mpc import CasADiNMPCController
from src.casadi_dynamics import CASADI_DYNAMICS
from ncp import NCPController
from ncp.algorithm.result import NCPResult
from ncp.utils.wrapping import wrap_angles


class NearestNeighborNCPController:
    """NCP controller with nearest-neighbor fallback for uncovered states.

    When a state falls inside a verified cell, behaves identically to NCPController.
    When a state is NOT in any verified cell, uses the control from the nearest
    verified cell center (L-inf distance).
    """
    def __init__(self, result: NCPResult, precision: float = 0.05, sentinel: float = 1337.0):
        self.inner = NCPController(result, sentinel=sentinel)
        self.centers = result.verified_centers
        self.radii = result.verified_radii
        self.controls = result.verified_controls
        self.indices = result.verified_indices
        self.sentinel = sentinel
        self.precision = precision

    @torch.no_grad()
    def lookup(self, states: torch.Tensor):
        # Try exact lookup first
        ctrl, taus = self.inner.lookup(states)

        # For states with tau=0 (no exact match), check if near origin first
        miss = taus == 0
        if miss.any() and self.centers.shape[0] > 0:
            miss_states = states[miss]  # (n_miss, D)

            # States inside the precision ball around origin: leave as tau=0 (already converged)
            at_origin = miss_states.abs().max(dim=-1).values < self.precision

            # Only apply NN for states that are NOT at the origin
            need_nn = miss.clone()
            need_nn_indices = miss.nonzero(as_tuple=True)[0]
            need_nn[need_nn_indices[at_origin]] = False

            if need_nn.any():
                nn_states = states[need_nn]
                # L-inf distance to each center
                dists = (nn_states[:, None, :] - self.centers[None, :, :]).abs().max(dim=-1).values  # (n, m)
                nn_idx = dists.argmin(dim=1)

                nn_ctrl = self.controls[nn_idx]
                nn_taus = self.indices[nn_idx].long()

                if nn_ctrl.dim() == 2:
                    from ncp.utils.tensor_ops import pad_tensor_rows_1d
                    nn_ctrl = pad_tensor_rows_1d(nn_ctrl, nn_taus, self.sentinel)

                ctrl[need_nn] = nn_ctrl
                taus[need_nn] = nn_taus

        return ctrl, taus


def load_ncp_result(path: Path) -> NCPResult:
    """Load NCPResult from saved dict."""
    data = torch.load(path, weights_only=True)
    return NCPResult(
        verified_centers=data["verified_centers"],
        verified_radii=data["verified_radii"],
        verified_controls=data["verified_controls"],
        verified_alphas=data["verified_alphas"],
        verified_indices=data["verified_indices"],
        unverified_centers=data.get("unverified_centers"),
        unverified_radii=data.get("unverified_radii"),
        elapsed_seconds=data.get("elapsed_seconds", 0.0),
    )


def simulate_ncp_native(controller, sys_info, ics, dt, max_steps, precision):
    """NCP native mode: apply full stored control sequence, then re-query."""
    device = ics.device
    B = ics.shape[0]
    state_dim = sys_info.state_spec.dim

    all_trajs = []
    all_ctrls = []
    converged = torch.zeros(B, dtype=torch.bool)
    steps = torch.zeros(B, dtype=torch.long)
    solve_times = []

    for i in range(B):
        state = ics[i:i+1].clone()  # (1, D)
        traj = [state.squeeze(0).clone()]
        ctrls = []
        step_count = 0

        while step_count < max_steps:
            t0 = time.time()
            ctrl, taus = controller.lookup(state)
            st = time.time() - t0
            solve_times.append(st)

            tau_steps = taus[0].item()
            if tau_steps == 0:
                # State not in any verified cell; stop
                steps[i] = step_count
                break

            # Apply control sequence for tau_steps
            for t in range(tau_steps):
                if step_count >= max_steps:
                    break
                if sys_info.control_spec.dim == 1:
                    u_t = ctrl[0, t:t+1]  # (1,)
                else:
                    u_t = ctrl[0, t:t+1]  # (1, ctrl_dim)

                dx = sys_info.dynamics_fn(state, u_t)
                state = state + dx * dt
                if sys_info.state_spec.wrap_dims:
                    state = wrap_angles(state, sys_info.state_spec.wrap_dims)

                traj.append(state.squeeze(0).clone())
                ctrls.append(u_t.squeeze(0).clone())
                step_count += 1

                # Check convergence
                if state.abs().max().item() < precision:
                    converged[i] = True
                    steps[i] = step_count
                    break

            if converged[i]:
                break

        if not converged[i]:
            steps[i] = step_count

        all_trajs.append(torch.stack(traj))
        all_ctrls.append(torch.stack(ctrls) if ctrls else torch.zeros(0, sys_info.control_spec.dim))

    return all_trajs, all_ctrls, converged, steps, solve_times


def simulate_ncp_step(controller, sys_info, ics, dt, max_steps, precision):
    """NCP step-by-step mode: apply one control, re-query every step."""
    B = ics.shape[0]

    all_trajs = []
    all_ctrls = []
    converged = torch.zeros(B, dtype=torch.bool)
    steps = torch.zeros(B, dtype=torch.long)
    solve_times = []

    for i in range(B):
        state = ics[i:i+1].clone()
        traj = [state.squeeze(0).clone()]
        ctrls = []

        for step in range(max_steps):
            t0 = time.time()
            ctrl, taus = controller.lookup(state)
            st = time.time() - t0
            solve_times.append(st)

            if taus[0].item() == 0:
                steps[i] = step
                break

            # Apply only first control
            if sys_info.control_spec.dim == 1:
                u_t = ctrl[0, 0:1]
            else:
                u_t = ctrl[0, 0:1]

            dx = sys_info.dynamics_fn(state, u_t)
            state = state + dx * dt
            if sys_info.state_spec.wrap_dims:
                state = wrap_angles(state, sys_info.state_spec.wrap_dims)

            traj.append(state.squeeze(0).clone())
            ctrls.append(u_t.squeeze(0).clone())

            if state.abs().max().item() < precision:
                converged[i] = True
                steps[i] = step + 1
                break
        else:
            steps[i] = max_steps

        all_trajs.append(torch.stack(traj))
        all_ctrls.append(torch.stack(ctrls) if ctrls else torch.zeros(0, sys_info.control_spec.dim))

    return all_trajs, all_ctrls, converged, steps, solve_times


def simulate_mppi(controller, sys_info, ics, dt, max_steps, precision):
    """MPPI-MPC simulation."""
    B = ics.shape[0]

    all_trajs = []
    all_ctrls = []
    converged_list = []
    steps_list = []
    solve_times = []

    for i in range(B):
        controller.reset()
        state = ics[i].clone()
        traj = [state.clone()]
        ctrls = []
        conv = False

        for step in range(max_steps):
            u, info = controller.step(state)
            solve_times.append(info["solve_time"])

            # Apply control
            x = state.unsqueeze(0)
            u_t = u.unsqueeze(0)
            if sys_info.control_spec.dim == 1 and u_t.dim() == 1:
                u_t = u_t.unsqueeze(-1)
            dx = sys_info.dynamics_fn(x, u_t)
            state = (x + dx * dt).squeeze(0)
            if sys_info.state_spec.wrap_dims:
                state = wrap_angles(state.unsqueeze(0), sys_info.state_spec.wrap_dims).squeeze(0)

            traj.append(state.clone())
            ctrls.append(u.clone())

            if state.abs().max().item() < precision:
                conv = True
                steps_list.append(step + 1)
                break
        else:
            steps_list.append(max_steps)

        converged_list.append(conv)
        all_trajs.append(torch.stack(traj))
        all_ctrls.append(torch.stack(ctrls) if ctrls else torch.zeros(0))

    return (all_trajs, all_ctrls, torch.tensor(converged_list),
            torch.tensor(steps_list), solve_times)


def simulate_casadi(controller, sys_info, ics, dt, max_steps, precision):
    """CasADi NMPC simulation."""
    B = ics.shape[0]
    nu = sys_info.control_spec.dim

    all_trajs = []
    all_ctrls = []
    converged_list = []
    steps_list = []
    solve_times = []
    all_ipopt_iters = []

    for i in range(B):
        controller.reset()
        state = ics[i].numpy().copy()
        traj = [state.copy()]
        ctrls = []
        conv = False

        for step in range(max_steps):
            u, info = controller.step(state)
            solve_times.append(info["solve_time"])
            all_ipopt_iters.append(info["ipopt_iterations"])

            # Apply control with PyTorch dynamics
            x_t = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
            u_t = torch.tensor(u.reshape(1, -1) if nu > 1 else u.reshape(1), dtype=torch.float32)
            dx = sys_info.dynamics_fn(x_t, u_t)
            state = (x_t + dx * dt).squeeze(0).numpy()

            if sys_info.state_spec.wrap_dims:
                for d in sys_info.state_spec.wrap_dims:
                    state[d] = ((state[d] + np.pi) % (2 * np.pi)) - np.pi

            traj.append(state.copy())
            ctrls.append(u.copy())

            if np.max(np.abs(state)) < precision:
                conv = True
                steps_list.append(step + 1)
                break
        else:
            steps_list.append(max_steps)

        converged_list.append(conv)
        all_trajs.append(torch.tensor(np.array(traj), dtype=torch.float32))
        all_ctrls.append(torch.tensor(np.array(ctrls), dtype=torch.float32) if ctrls
                         else torch.zeros(0, nu))

    return (all_trajs, all_ctrls, torch.tensor(converged_list),
            torch.tensor(steps_list), solve_times)


def pad_and_stack(trajs, max_len, dim):
    """Pad variable-length trajectories to common length and stack."""
    padded = []
    for t in trajs:
        if t.shape[0] < max_len:
            pad = t[-1:].expand(max_len - t.shape[0], -1)
            t = torch.cat([t, pad], dim=0)
        else:
            t = t[:max_len]
        padded.append(t)
    return torch.stack(padded)


def save_closed_loop_metrics(
    trajs, ctrls, converged, steps, solve_times, total_time,
    system_name, controller_type, out_dir,
):
    """Compute and save closed-loop metrics."""
    B = len(trajs)
    max_traj_len = max(t.shape[0] for t in trajs)
    state_dim = trajs[0].shape[-1]

    traj_tensor = pad_and_stack(trajs, max_traj_len, state_dim)

    if ctrls and ctrls[0].numel() > 0:
        ctrl_dim = ctrls[0].shape[-1] if ctrls[0].dim() > 1 else 1
        max_ctrl_len = max(c.shape[0] for c in ctrls) if any(c.numel() > 0 for c in ctrls) else 1
        ctrl_tensor = pad_and_stack(
            [c if c.dim() > 1 else c.unsqueeze(-1) for c in ctrls if c.numel() > 0],
            max_ctrl_len, ctrl_dim,
        ) if any(c.numel() > 0 for c in ctrls) else torch.zeros(B, 1, 1)
    else:
        ctrl_tensor = torch.zeros(B, 1, 1)

    metrics = compute_closed_loop_metrics(
        trajectories=traj_tensor,
        controls=ctrl_tensor,
        converged=converged,
        steps=steps,
        solve_times=solve_times,
        total_wall_time=total_time,
        system_name=system_name,
        controller_type=controller_type,
    )
    metrics.save(out_dir / f"{system_name}_{controller_type}_metrics.json")

    # Also save raw simulation data
    torch.save({
        "trajectories": trajs,
        "controls": ctrls,
        "converged": converged,
        "steps": steps,
        "solve_times": solve_times,
    }, out_dir / f"{system_name}_{controller_type}_raw.pt")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Closed-loop evaluation of all controllers")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--system", type=str, help="System name")
    group.add_argument("--index", type=int, help="SLURM array index (0-14)")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--controllers", nargs="+",
                        default=["ncp_native", "ncp_step", "ncp_nn_native", "ncp_nn_step", "mppi_mpc", "casadi_mpc"],
                        help="Which controllers to evaluate")
    args = parser.parse_args()

    if args.system:
        sys_info = get_system(args.system)
    else:
        sys_info = get_system_by_index(args.index)

    name = sys_info.name
    cfg = load_config(name, args.phase)

    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(__file__).parent.parent / "results" / f"phase{args.phase}"

    cl_dir = base_dir / "closed_loop"
    cl_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Closed-Loop Evaluation for {name} (phase {args.phase}) ===")
    print(f"  Controllers: {args.controllers}")
    print(f"  Max steps: {cfg.sim.max_steps}, Precision: {cfg.sim.precision}")

    # Load ICs
    ic_path = base_dir / "initial_conditions" / f"{name}_ics.pt"
    if not ic_path.exists():
        print(f"ERROR: ICs not found at {ic_path}. Run NCP or MPC first.")
        sys.exit(1)
    ics = load_ics(ic_path)
    print(f"  ICs: {ics.shape[0]} states")

    results = {}

    # --- NCP controllers ---
    if any(c.startswith("ncp") for c in args.controllers):
        ncp_path = base_dir / "ncp" / f"{name}_result.pt"
        if not ncp_path.exists():
            print(f"WARNING: NCP result not found at {ncp_path}, skipping NCP controllers")
        else:
            ncp_result = load_ncp_result(ncp_path)
            controller = NCPController(ncp_result)

            if "ncp_native" in args.controllers:
                print(f"\n--- NCP Native Mode ---")
                t0 = time.time()
                trajs, ctrls, conv, steps, st = simulate_ncp_native(
                    controller, sys_info, ics, cfg.ncp.dt, cfg.sim.max_steps, cfg.sim.precision)
                total = time.time() - t0
                m = save_closed_loop_metrics(trajs, ctrls, conv, steps, st, total, name, "ncp_native", cl_dir)
                results["ncp_native"] = m
                print(f"  Convergence: {m.convergence_rate:.1%}, Time: {total:.1f}s")

            if "ncp_step" in args.controllers:
                print(f"\n--- NCP Step-by-Step Mode ---")
                t0 = time.time()
                trajs, ctrls, conv, steps, st = simulate_ncp_step(
                    controller, sys_info, ics, cfg.ncp.dt, cfg.sim.max_steps, cfg.sim.precision)
                total = time.time() - t0
                m = save_closed_loop_metrics(trajs, ctrls, conv, steps, st, total, name, "ncp_step", cl_dir)
                results["ncp_step"] = m
                print(f"  Convergence: {m.convergence_rate:.1%}, Time: {total:.1f}s")

            # Nearest-neighbor NCP modes
            nn_controller = NearestNeighborNCPController(ncp_result, precision=cfg.sim.precision)

            if "ncp_nn_native" in args.controllers:
                print(f"\n--- NCP Nearest-Neighbor Native Mode ---")
                t0 = time.time()
                trajs, ctrls, conv, steps, st = simulate_ncp_native(
                    nn_controller, sys_info, ics, cfg.ncp.dt, cfg.sim.max_steps, cfg.sim.precision)
                total = time.time() - t0
                m = save_closed_loop_metrics(trajs, ctrls, conv, steps, st, total, name, "ncp_nn_native", cl_dir)
                results["ncp_nn_native"] = m
                print(f"  Convergence: {m.convergence_rate:.1%}, Time: {total:.1f}s")

            if "ncp_nn_step" in args.controllers:
                print(f"\n--- NCP Nearest-Neighbor Step Mode ---")
                t0 = time.time()
                trajs, ctrls, conv, steps, st = simulate_ncp_step(
                    nn_controller, sys_info, ics, cfg.ncp.dt, cfg.sim.max_steps, cfg.sim.precision)
                total = time.time() - t0
                m = save_closed_loop_metrics(trajs, ctrls, conv, steps, st, total, name, "ncp_nn_step", cl_dir)
                results["ncp_nn_step"] = m
                print(f"  Convergence: {m.convergence_rate:.1%}, Time: {total:.1f}s")

    # --- MPPI-MPC ---
    if "mppi_mpc" in args.controllers:
        print(f"\n--- MPPI-MPC ---")
        mppi_ctrl = MPPIMPCController(
            dynamics_fn=sys_info.dynamics_fn,
            state_dim=sys_info.state_spec.dim,
            ctrl_dim=sys_info.control_spec.dim,
            control_lower=sys_info.control_spec.lower_bounds,
            control_upper=sys_info.control_spec.upper_bounds,
            dt=cfg.ncp.dt,
            horizon_steps=cfg.mpc.horizon_steps,
            num_samples=cfg.mpc.num_samples,
            sigma=cfg.mpc.sigma,
            temperature=cfg.mpc.temperature,
            Q_diag=cfg.mpc.Q_diag,
            R_diag=cfg.mpc.R_diag,
            wrap_dims=sys_info.state_spec.wrap_dims,
        )
        t0 = time.time()
        trajs, ctrls, conv, steps, st = simulate_mppi(
            mppi_ctrl, sys_info, ics, cfg.ncp.dt, cfg.sim.max_steps, cfg.sim.precision)
        total = time.time() - t0
        m = save_closed_loop_metrics(trajs, ctrls, conv, steps, st, total, name, "mppi_mpc", cl_dir)
        results["mppi_mpc"] = m
        print(f"  Convergence: {m.convergence_rate:.1%}, Time: {total:.1f}s")

    # --- CasADi NMPC ---
    if "casadi_mpc" in args.controllers:
        print(f"\n--- CasADi NMPC ---")
        casadi_dyn_fn = CASADI_DYNAMICS[name]
        nx = sys_info.state_spec.dim
        nu = sys_info.control_spec.dim

        Q_diag = np.array(cfg.casadi.Q_diag)
        R_diag = np.array(cfg.casadi.R_diag)
        if len(Q_diag) == 1:
            Q_diag = np.ones(nx) * Q_diag[0]
        if len(R_diag) == 1:
            R_diag = np.ones(nu) * R_diag[0]
        Qf_diag = Q_diag * cfg.casadi.terminal_weight

        casadi_ctrl = CasADiNMPCController(
            casadi_dynamics_fn=casadi_dyn_fn,
            state_dim=nx,
            ctrl_dim=nu,
            control_lower=np.array(sys_info.control_spec.lower_bounds.tolist()),
            control_upper=np.array(sys_info.control_spec.upper_bounds.tolist()),
            dt=cfg.ncp.dt,
            horizon_steps=cfg.casadi.horizon_steps,
            Q_diag=Q_diag,
            R_diag=R_diag,
            Qf_diag=Qf_diag,
            max_iter=cfg.casadi.max_iter,
            tol=cfg.casadi.tol,
        )
        t0 = time.time()
        trajs, ctrls, conv, steps, st = simulate_casadi(
            casadi_ctrl, sys_info, ics, cfg.ncp.dt, cfg.sim.max_steps, cfg.sim.precision)
        total = time.time() - t0
        m = save_closed_loop_metrics(trajs, ctrls, conv, steps, st, total, name, "casadi_mpc", cl_dir)
        results["casadi_mpc"] = m
        print(f"  Convergence: {m.convergence_rate:.1%}, Time: {total:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {name} (phase {args.phase})")
    print(f"{'='*60}")
    print(f"{'Controller':<18} {'Conv%':>8} {'Steps(mean)':>12} {'SolveTime(ms)':>14} {'TrajCost':>10}")
    print(f"{'-'*65}")
    for cname, m in results.items():
        print(f"{cname:<18} {m.convergence_rate:>7.1%} {m.steps_to_converge_mean:>12.1f} "
              f"{m.solve_time_mean*1000:>14.2f} {m.trajectory_cost:>10.2f}")


if __name__ == "__main__":
    main()
