#!/usr/bin/env python3
"""Collect results from all systems into summary CSV and JSON."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import SYSTEM_ORDER
from src.metrics import NCPMetrics, MPCMetrics, ClosedLoopMetrics


def main():
    parser = argparse.ArgumentParser(description="Aggregate results into summary tables")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    if args.results_dir:
        base_dir = Path(args.results_dir)
    else:
        base_dir = Path(__file__).parent.parent / "results" / f"phase{args.phase}"

    ncp_dir = base_dir / "ncp"
    mppi_dir = base_dir / "mppi_mpc"
    casadi_dir = base_dir / "casadi_mpc"
    cl_dir = base_dir / "closed_loop"

    controller_types = ["ncp_native", "ncp_step", "mppi_mpc", "casadi_mpc"]

    # Collect NCP build metrics
    ncp_summary = []
    for name in SYSTEM_ORDER:
        path = ncp_dir / f"{name}_metrics.json"
        if path.exists():
            m = NCPMetrics.load(path)
            ncp_summary.append(vars(m))
        else:
            ncp_summary.append({"system_name": name, "status": "missing"})

    # Collect MPC metrics
    mpc_summary = []
    for mpc_type, mpc_dir_path in [("mppi", mppi_dir), ("casadi", casadi_dir)]:
        for name in SYSTEM_ORDER:
            path = mpc_dir_path / f"{name}_metrics.json"
            if path.exists():
                m = MPCMetrics.load(path)
                mpc_summary.append(vars(m))
            else:
                mpc_summary.append({"system_name": name, "controller_type": mpc_type, "status": "missing"})

    # Collect closed-loop metrics
    cl_summary = []
    for name in SYSTEM_ORDER:
        row = {"system_name": name}
        for ctype in controller_types:
            path = cl_dir / f"{name}_{ctype}_metrics.json"
            if path.exists():
                m = ClosedLoopMetrics.load(path)
                for k, v in vars(m).items():
                    if k not in ("system_name", "controller_type"):
                        row[f"{ctype}_{k}"] = v
            else:
                row[f"{ctype}_status"] = "missing"
        cl_summary.append(row)

    # Write CSVs
    output_dir = base_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    # NCP build summary
    if ncp_summary:
        with open(output_dir / "ncp_build_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ncp_summary[0].keys())
            writer.writeheader()
            writer.writerows(ncp_summary)

    # MPC summary
    if mpc_summary:
        with open(output_dir / "mpc_summary.csv", "w", newline="") as f:
            keys = set()
            for row in mpc_summary:
                keys.update(row.keys())
            writer = csv.DictWriter(f, fieldnames=sorted(keys))
            writer.writeheader()
            writer.writerows(mpc_summary)

    # Closed-loop comparison
    if cl_summary:
        with open(output_dir / "closed_loop_summary.csv", "w", newline="") as f:
            keys = set()
            for row in cl_summary:
                keys.update(row.keys())
            ordered_keys = ["system_name"] + sorted(k for k in keys if k != "system_name")
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            writer.writerows(cl_summary)

    # Also write JSON for programmatic access
    all_results = {
        "phase": args.phase,
        "ncp_build": ncp_summary,
        "mpc": mpc_summary,
        "closed_loop": cl_summary,
    }
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*100}")
    print(f"PHASE {args.phase} SUMMARY")
    print(f"{'='*100}")

    # NCP build results
    print(f"\n--- NCP Build Results ---")
    print(f"{'System':<16} {'Build(s)':>10} {'Verified':>10} {'Unverified':>10} {'Coverage':>10} {'Alpha(mean)':>12}")
    for row in ncp_summary:
        if "status" in row:
            print(f"{row['system_name']:<16} {'MISSING':>10}")
        else:
            print(f"{row['system_name']:<16} {row['build_time_seconds']:>10.1f} "
                  f"{row['num_verified_cells']:>10} {row['num_unverified_cells']:>10} "
                  f"{row['coverage_ratio']:>10.3f} {row['alpha_mean']:>12.4f}")

    # Closed-loop comparison
    print(f"\n--- Closed-Loop Convergence Rates ---")
    header = f"{'System':<16}"
    for ctype in controller_types:
        header += f" {ctype:>14}"
    print(header)
    for row in cl_summary:
        line = f"{row['system_name']:<16}"
        for ctype in controller_types:
            key = f"{ctype}_convergence_rate"
            if key in row:
                line += f" {row[key]:>13.1%}"
            else:
                line += f" {'N/A':>14}"
        print(line)

    # Solve time comparison
    print(f"\n--- Mean Solve Time (ms) ---")
    header = f"{'System':<16}"
    for ctype in controller_types:
        header += f" {ctype:>14}"
    print(header)
    for row in cl_summary:
        line = f"{row['system_name']:<16}"
        for ctype in controller_types:
            key = f"{ctype}_solve_time_mean"
            if key in row:
                line += f" {row[key]*1000:>13.2f}"
            else:
                line += f" {'N/A':>14}"
        print(line)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
