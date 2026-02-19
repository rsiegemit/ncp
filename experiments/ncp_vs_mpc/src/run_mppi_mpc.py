#!/usr/bin/env python3
"""CLI: Run MPPI-MPC simulation for one system, save metrics."""

import argparse
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_system, get_system_by_index
from src.config import load_config
from src.initial_conditions import generate_ics, save_ics, load_ics
from src.metrics import MPCMetrics
from src.mppi_mpc import MPPIMPCController


def main():
    parser = argparse.ArgumentParser(description="Run MPPI-MPC for one system")
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

    mpc_dir = out_dir / "mppi_mpc"
    ic_dir = out_dir / "initial_conditions"
    mpc_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running MPPI-MPC for {name} (phase {args.phase}) ===")

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

    horizon_steps = cfg.mpc.horizon_steps
    print(f"Config: horizon={horizon_steps}, K={cfg.mpc.num_samples}, "
          f"lambda={cfg.mpc.temperature}, sigma={cfg.mpc.sigma}")
    print(f"  Q_diag={cfg.mpc.Q_diag}, R_diag={cfg.mpc.R_diag}")
    print(f"  dt={cfg.ncp.dt}, max_steps={cfg.sim.max_steps}")

    # Create controller with proper quadratic cost
    controller = MPPIMPCController(
        dynamics_fn=sys_info.dynamics_fn,
        state_dim=sys_info.state_spec.dim,
        ctrl_dim=sys_info.control_spec.dim,
        control_lower=sys_info.control_spec.lower_bounds,
        control_upper=sys_info.control_spec.upper_bounds,
        dt=cfg.ncp.dt,
        horizon_steps=horizon_steps,
        num_samples=cfg.mpc.num_samples,
        sigma=cfg.mpc.sigma,
        temperature=cfg.mpc.temperature,
        Q_diag=cfg.mpc.Q_diag,
        R_diag=cfg.mpc.R_diag,
        wrap_dims=sys_info.state_spec.wrap_dims,
    )

    # Run simulation for each IC
    all_solve_times = []
    all_trajectories = []
    all_controls = []
    all_converged = []
    all_steps = []

    total_t0 = time.time()

    for i, ic in enumerate(ics):
        controller.reset()
        state = ic.clone()
        traj = [state.clone()]
        ctrls = []
        converged = False

        for step in range(cfg.sim.max_steps):
            u, info = controller.step(state)
            all_solve_times.append(info["solve_time"])

            # Apply control to get next state (Euler step)
            x = state.unsqueeze(0)
            u_t = u.unsqueeze(0)
            if sys_info.control_spec.dim == 1 and u_t.dim() == 1:
                u_t = u_t.unsqueeze(-1)
            dx = sys_info.dynamics_fn(x, u_t)
            state = (x + dx * cfg.ncp.dt).squeeze(0)

            # Wrap angles if needed
            if sys_info.state_spec.wrap_dims:
                from ncp.utils.wrapping import wrap_angles
                state = wrap_angles(state.unsqueeze(0), sys_info.state_spec.wrap_dims).squeeze(0)

            traj.append(state.clone())
            ctrls.append(u.clone())

            # Check convergence
            if state.abs().max().item() < cfg.sim.precision:
                converged = True
                all_steps.append(step + 1)
                break
        else:
            all_steps.append(cfg.sim.max_steps)

        all_converged.append(converged)
        all_trajectories.append(torch.stack(traj))
        all_controls.append(torch.stack(ctrls) if ctrls else torch.zeros(0))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  IC {i+1}/{len(ics)}: {'converged' if converged else 'not converged'} "
                  f"in {all_steps[-1]} steps, last solve {all_solve_times[-1]*1000:.1f}ms")

    total_time = time.time() - total_t0

    # Save raw trajectories and controls
    torch.save({
        "trajectories": all_trajectories,
        "controls": all_controls,
        "converged": torch.tensor(all_converged),
        "steps": torch.tensor(all_steps),
        "solve_times": all_solve_times,
    }, mpc_dir / f"{name}_raw.pt")

    # Compute and save metrics
    st = torch.tensor(all_solve_times)
    metrics = MPCMetrics(
        system_name=name,
        controller_type="mppi",
        total_sim_time_seconds=total_time,
        solve_time_mean=st.mean().item(),
        solve_time_median=st.median().item(),
        solve_time_max=st.max().item(),
    )
    metrics.save(mpc_dir / f"{name}_metrics.json")

    conv_rate = sum(all_converged) / len(all_converged)
    print(f"\n=== MPPI-MPC Results for {name} ===")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Convergence rate: {conv_rate:.1%}")
    print(f"  Solve time: mean={st.mean().item()*1000:.2f}ms, "
          f"median={st.median().item()*1000:.2f}ms, max={st.max().item()*1000:.2f}ms")


if __name__ == "__main__":
    main()
