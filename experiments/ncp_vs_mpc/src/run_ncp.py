#!/usr/bin/env python3
"""CLI: Build NCP for one system, save result and metrics."""

import argparse
import sys
import time
import torch
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_system, get_system_by_index, SYSTEM_ORDER
from src.config import load_config
from src.initial_conditions import generate_ics, save_ics
from src.metrics import NCPMetrics
from ncp import NCPBuilder


def main():
    parser = argparse.ArgumentParser(description="Build NCP for one system")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--system", type=str, help="System name")
    group.add_argument("--index", type=int, help="SLURM array index (0-14)")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Resolve system
    if args.system:
        sys_info = get_system(args.system)
    else:
        sys_info = get_system_by_index(args.index)

    name = sys_info.name
    cfg = load_config(name, args.phase)

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent.parent / "results" / f"phase{args.phase}"

    ncp_dir = out_dir / "ncp"
    ic_dir = out_dir / "initial_conditions"
    ncp_dir.mkdir(parents=True, exist_ok=True)
    ic_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building NCP for {name} (phase {args.phase}) ===")
    print(f"Config: R={cfg.ncp.radius}, eps={cfg.ncp.epsilon}, tau={cfg.ncp.tau}, "
          f"dt={cfg.ncp.dt}, samples={cfg.ncp.num_samples}, splits={cfg.ncp.max_splits}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Checkpoint path for intermediate saves
    result_path = ncp_dir / f"{name}_result.pt"
    checkpoint_count = [0]

    def save_result(res):
        """Save an NCPResult to disk."""
        torch.save({
            "verified_centers": res.verified_centers.cpu(),
            "verified_radii": res.verified_radii.cpu(),
            "verified_controls": res.verified_controls.cpu(),
            "verified_alphas": res.verified_alphas.cpu(),
            "verified_indices": res.verified_indices.cpu(),
            "unverified_centers": res.unverified_centers.cpu() if res.unverified_centers is not None else None,
            "unverified_radii": res.unverified_radii.cpu() if res.unverified_radii is not None else None,
            "elapsed_seconds": res.elapsed_seconds,
        }, result_path)

    def checkpoint_fn(partial_result):
        checkpoint_count[0] += 1
        n_verified = partial_result.verified_centers.shape[0]
        print(f"  [checkpoint {checkpoint_count[0]}] {n_verified} verified cells, "
              f"{partial_result.elapsed_seconds:.0f}s elapsed", flush=True)
        save_result(partial_result)

    # Build NCP
    builder = NCPBuilder(
        env_factory=sys_info.env_class,
        state_spec=sys_info.state_spec,
        control_spec=sys_info.control_spec,
        dynamics_fn=sys_info.dynamics_fn,
        radius=cfg.ncp.radius,
        epsilon=cfg.ncp.epsilon,
        lipschitz=cfg.ncp.lipschitz,
        tau=cfg.ncp.tau,
        dt=cfg.ncp.dt,
        min_alpha=cfg.ncp.min_alpha,
        max_splits=cfg.ncp.max_splits,
        num_samples=cfg.ncp.num_samples,
        batch_size=cfg.ncp.batch_size,
        reuse=cfg.ncp.reuse,
    )

    t0 = time.time()
    result = builder.build(
        checkpoint_fn=checkpoint_fn,
        checkpoint_interval=300.0,  # save every 5 minutes
    )
    build_time = time.time() - t0

    print(f"Build complete in {build_time:.1f}s")
    print(f"  Verified cells: {result.verified_centers.shape[0]}")
    if result.unverified_centers is not None:
        print(f"  Unverified cells: {result.unverified_centers.shape[0]}")

    # Save final NCP result
    save_result(result)
    print(f"Saved result to {result_path}")

    # Compute and save metrics
    n_verified = result.verified_centers.shape[0]
    n_unverified = result.unverified_centers.shape[0] if result.unverified_centers is not None else 0
    total_cells = n_verified + n_unverified

    alphas = result.verified_alphas
    metrics = NCPMetrics(
        system_name=name,
        build_time_seconds=build_time,
        num_verified_cells=n_verified,
        num_unverified_cells=n_unverified,
        coverage_ratio=n_verified / total_cells if total_cells > 0 else 0.0,
        alpha_mean=alphas.mean().item() if n_verified > 0 else 0.0,
        alpha_median=alphas.median().item() if n_verified > 0 else 0.0,
        alpha_min=alphas.min().item() if n_verified > 0 else 0.0,
        alpha_max=alphas.max().item() if n_verified > 0 else 0.0,
    )
    metrics.save(ncp_dir / f"{name}_metrics.json")
    print(f"Saved metrics to {ncp_dir / f'{name}_metrics.json'}")

    # Generate and save ICs for this system (shared with MPC controllers)
    ics = generate_ics(
        state_dim=sys_info.state_spec.dim,
        radius=cfg.ncp.radius,
        grid_per_dim=cfg.ic.grid_per_dim,
        num_random=cfg.ic.num_random,
        wrap_dims=sys_info.state_spec.wrap_dims,
        seed=cfg.ic.seed,
    )
    save_ics(ics, ic_dir / f"{name}_ics.pt")
    print(f"Saved {ics.shape[0]} ICs to {ic_dir / f'{name}_ics.pt'}")


if __name__ == "__main__":
    main()
