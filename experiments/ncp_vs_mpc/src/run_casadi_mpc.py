#!/usr/bin/env python3
"""CLI: Run CasADi/IPOPT NMPC simulation for one system, save metrics."""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_system, get_system_by_index
from src.config import load_config
from src.initial_conditions import generate_ics, save_ics, load_ics
from src.metrics import MPCMetrics
from src.casadi_mpc import CasADiNMPCController
from src.casadi_dynamics import CASADI_DYNAMICS


def main():
    parser = argparse.ArgumentParser(description="Run CasADi NMPC for one system")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--system", type=str, help="System name")
    group.add_argument("--index", type=int, help="SLURM array index (0-14)")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.system:
        sys_info = get_system(args.system)
    else:
        sys_info = get_system_by_index(args.index)

    name = sys_info.name
    cfg = load_config(name, args.phase)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent.parent / "results" / f"phase{args.phase}"

    casadi_dir = out_dir / "casadi_mpc"
    ic_dir = out_dir / "initial_conditions"
    casadi_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running CasADi NMPC for {name} (phase {args.phase}) ===")

    # Load or generate ICs
    ic_path = ic_dir / f"{name}_ics.pt"
    if ic_path.exists():
        ics = load_ics(ic_path)
        print(f"Loaded {ics.shape[0]} ICs from {ic_path}")
    else:
        ics = generate_ics(
            state_dim=sys_info.state_spec.dim,
            radius=cfg.ncp.radius,
            grid_per_dim=cfg.ic.grid_per_dim,
            num_random=cfg.ic.num_random,
            wrap_dims=sys_info.state_spec.wrap_dims,
            seed=cfg.ic.seed,
        )
        save_ics(ics, ic_path)
        print(f"Generated and saved {ics.shape[0]} ICs")

    horizon_steps = cfg.casadi.horizon_steps
    print(f"Config: horizon={horizon_steps}, dt={cfg.ncp.dt}, "
          f"max_steps={cfg.sim.max_steps}, Q={cfg.casadi.Q_diag}, R={cfg.casadi.R_diag}")

    # Get CasADi dynamics function
    casadi_dyn_fn = CASADI_DYNAMICS[name]

    # Build cost weight vectors from per-dimension config
    nx = sys_info.state_spec.dim
    nu = sys_info.control_spec.dim
    Q_diag = np.array(cfg.casadi.Q_diag)
    R_diag = np.array(cfg.casadi.R_diag)
    # Broadcast scalar to full dimension if needed
    if len(Q_diag) == 1:
        Q_diag = np.ones(nx) * Q_diag[0]
    if len(R_diag) == 1:
        R_diag = np.ones(nu) * R_diag[0]
    Qf_diag = Q_diag * cfg.casadi.terminal_weight

    # Create controller
    controller = CasADiNMPCController(
        casadi_dynamics_fn=casadi_dyn_fn,
        state_dim=nx,
        ctrl_dim=nu,
        control_lower=np.array(sys_info.control_spec.lower_bounds.tolist()),
        control_upper=np.array(sys_info.control_spec.upper_bounds.tolist()),
        dt=cfg.ncp.dt,
        horizon_steps=horizon_steps,
        Q_diag=Q_diag,
        R_diag=R_diag,
        Qf_diag=Qf_diag,
        max_iter=cfg.casadi.max_iter,
        tol=cfg.casadi.tol,
    )

    # Run simulation for each IC
    all_solve_times = []
    all_ipopt_iters = []
    all_trajectories = []
    all_controls = []
    all_converged = []
    all_steps = []
    convergence_failures = 0

    total_t0 = time.time()

    for i, ic in enumerate(ics):
        controller.reset()
        state = ic.numpy().copy()
        traj = [state.copy()]
        ctrls = []
        converged = False

        for step in range(cfg.sim.max_steps):
            u, info = controller.step(state)
            all_solve_times.append(info["solve_time"])
            all_ipopt_iters.append(info["ipopt_iterations"])
            if not info["converged"]:
                convergence_failures += 1

            # Apply control: Euler step (matching NCP dynamics)
            x_np = state.reshape(1, -1)
            u_np = u.reshape(1, -1) if nu > 1 else u.reshape(1)

            # Use PyTorch dynamics for ground-truth stepping
            x_t = torch.tensor(x_np, dtype=torch.float32)
            u_t = torch.tensor(u_np, dtype=torch.float32)
            dx = sys_info.dynamics_fn(x_t, u_t)
            state = (x_t + dx * cfg.ncp.dt).squeeze(0).numpy()

            # Wrap angles if needed
            if sys_info.state_spec.wrap_dims:
                for d in sys_info.state_spec.wrap_dims:
                    state[d] = ((state[d] + np.pi) % (2 * np.pi)) - np.pi

            traj.append(state.copy())
            ctrls.append(u.copy())

            # Check convergence
            if np.max(np.abs(state)) < cfg.sim.precision:
                converged = True
                all_steps.append(step + 1)
                break
        else:
            all_steps.append(cfg.sim.max_steps)

        all_converged.append(converged)
        all_trajectories.append(np.array(traj))
        all_controls.append(np.array(ctrls) if ctrls else np.zeros((0, nu)))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  IC {i+1}/{len(ics)}: {'converged' if converged else 'not converged'} "
                  f"in {all_steps[-1]} steps, last solve {all_solve_times[-1]*1000:.1f}ms")

    total_time = time.time() - total_t0

    # Save raw results
    torch.save({
        "trajectories": [torch.tensor(t, dtype=torch.float32) for t in all_trajectories],
        "controls": [torch.tensor(c, dtype=torch.float32) for c in all_controls],
        "converged": torch.tensor(all_converged),
        "steps": torch.tensor(all_steps),
        "solve_times": all_solve_times,
        "ipopt_iterations": all_ipopt_iters,
    }, casadi_dir / f"{name}_raw.pt")

    # Compute and save metrics
    st = torch.tensor(all_solve_times)
    metrics = MPCMetrics(
        system_name=name,
        controller_type="casadi",
        total_sim_time_seconds=total_time,
        solve_time_mean=st.mean().item(),
        solve_time_median=st.median().item(),
        solve_time_max=st.max().item(),
        ipopt_iterations_mean=np.mean(all_ipopt_iters),
        convergence_failures=convergence_failures,
    )
    metrics.save(casadi_dir / f"{name}_metrics.json")

    conv_rate = sum(all_converged) / len(all_converged)
    print(f"\n=== CasADi NMPC Results for {name} ===")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Convergence rate: {conv_rate:.1%}")
    print(f"  Solve time: mean={st.mean().item()*1000:.2f}ms, "
          f"median={st.median().item()*1000:.2f}ms, max={st.max().item()*1000:.2f}ms")
    print(f"  IPOPT iterations (mean): {np.mean(all_ipopt_iters):.1f}")
    print(f"  IPOPT convergence failures: {convergence_failures}")


if __name__ == "__main__":
    main()
