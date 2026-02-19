#!/usr/bin/env python3
"""Generate comparison figures from aggregated results."""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import SYSTEM_ORDER

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)


def load_results(base_dir: Path) -> dict:
    summary_path = base_dir / "summary" / "all_results.json"
    if not summary_path.exists():
        print(f"ERROR: Run aggregate_results.py first. Missing: {summary_path}")
        sys.exit(1)
    with open(summary_path) as f:
        return json.load(f)


def plot_convergence_rates(results: dict, output_dir: Path):
    """Bar chart: convergence rate per system, grouped by controller."""
    cl = results["closed_loop"]
    controllers = ["ncp_native", "ncp_step", "mppi_mpc", "casadi_mpc"]
    labels = ["NCP (native)", "NCP (step)", "MPPI-MPC", "CasADi NMPC"]
    colors = ["#2196F3", "#64B5F6", "#FF9800", "#4CAF50"]

    systems = [r["system_name"] for r in cl]
    x = np.arange(len(systems))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (ctrl, label, color) in enumerate(zip(controllers, labels, colors)):
        rates = []
        for row in cl:
            key = f"{ctrl}_convergence_rate"
            rates.append(row.get(key, 0.0))
        ax.bar(x + i * width, rates, width, label=label, color=color)

    ax.set_ylabel("Convergence Rate")
    ax.set_title("Closed-Loop Convergence Rate by System and Controller")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(systems, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "convergence_rates.png", dpi=150)
    plt.close()
    print(f"  Saved convergence_rates.png")


def plot_solve_times(results: dict, output_dir: Path):
    """Bar chart: mean solve time per system, grouped by controller."""
    cl = results["closed_loop"]
    controllers = ["ncp_native", "ncp_step", "mppi_mpc", "casadi_mpc"]
    labels = ["NCP (native)", "NCP (step)", "MPPI-MPC", "CasADi NMPC"]
    colors = ["#2196F3", "#64B5F6", "#FF9800", "#4CAF50"]

    systems = [r["system_name"] for r in cl]
    x = np.arange(len(systems))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (ctrl, label, color) in enumerate(zip(controllers, labels, colors)):
        times = []
        for row in cl:
            key = f"{ctrl}_solve_time_mean"
            val = row.get(key, 0.0)
            times.append(val * 1000 if val else 0.0)  # Convert to ms
        ax.bar(x + i * width, times, width, label=label, color=color)

    ax.set_ylabel("Mean Solve Time (ms)")
    ax.set_title("Per-Step Solve Time by System and Controller")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(systems, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "solve_times.png", dpi=150)
    plt.close()
    print(f"  Saved solve_times.png")


def plot_trajectory_cost(results: dict, output_dir: Path):
    """Bar chart: mean trajectory cost per system, grouped by controller."""
    cl = results["closed_loop"]
    controllers = ["ncp_native", "ncp_step", "mppi_mpc", "casadi_mpc"]
    labels = ["NCP (native)", "NCP (step)", "MPPI-MPC", "CasADi NMPC"]
    colors = ["#2196F3", "#64B5F6", "#FF9800", "#4CAF50"]

    systems = [r["system_name"] for r in cl]
    x = np.arange(len(systems))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (ctrl, label, color) in enumerate(zip(controllers, labels, colors)):
        costs = []
        for row in cl:
            key = f"{ctrl}_trajectory_cost"
            costs.append(row.get(key, 0.0))
        ax.bar(x + i * width, costs, width, label=label, color=color)

    ax.set_ylabel("Trajectory Cost (sum ||x||)")
    ax.set_title("Trajectory Cost by System and Controller")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(systems, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "trajectory_cost.png", dpi=150)
    plt.close()
    print(f"  Saved trajectory_cost.png")


def plot_ncp_coverage(results: dict, output_dir: Path):
    """Bar chart: NCP coverage and alpha distribution."""
    ncp = results["ncp_build"]
    valid = [r for r in ncp if "status" not in r]
    if not valid:
        return

    systems = [r["system_name"] for r in valid]
    coverage = [r["coverage_ratio"] for r in valid]
    alpha_mean = [r["alpha_mean"] for r in valid]
    alpha_min = [r["alpha_min"] for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(len(systems)), coverage, color="#2196F3")
    ax1.set_ylabel("Coverage Ratio")
    ax1.set_title("NCP Coverage by System")
    ax1.set_xticks(range(len(systems)))
    ax1.set_xticklabels(systems, rotation=45, ha="right")
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.3)

    x = np.arange(len(systems))
    ax2.bar(x - 0.15, alpha_mean, 0.3, label="Mean alpha", color="#FF9800")
    ax2.bar(x + 0.15, alpha_min, 0.3, label="Min alpha", color="#4CAF50")
    ax2.set_ylabel("Alpha (contraction rate)")
    ax2.set_title("NCP Contraction Rates")
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "ncp_coverage_alpha.png", dpi=150)
    plt.close()
    print(f"  Saved ncp_coverage_alpha.png")


def plot_build_vs_solve(results: dict, output_dir: Path):
    """Scatter: NCP build time vs total MPC solve time."""
    ncp = {r["system_name"]: r for r in results["ncp_build"] if "status" not in r}
    cl = {r["system_name"]: r for r in results["closed_loop"]}

    systems = []
    build_times = []
    mppi_times = []
    casadi_times = []

    for name in SYSTEM_ORDER:
        if name in ncp and name in cl:
            row = cl[name]
            bt = ncp[name]["build_time_seconds"]
            mppi_t = row.get("mppi_mpc_total_wall_time", None)
            casadi_t = row.get("casadi_mpc_total_wall_time", None)
            if mppi_t is not None and casadi_t is not None:
                systems.append(name)
                build_times.append(bt)
                mppi_times.append(mppi_t)
                casadi_times.append(casadi_t)

    if not systems:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(build_times, mppi_times, s=100, c="#FF9800", marker="o", label="MPPI-MPC total", zorder=5)
    ax.scatter(build_times, casadi_times, s=100, c="#4CAF50", marker="^", label="CasADi total", zorder=5)

    for i, name in enumerate(systems):
        ax.annotate(name, (build_times[i], mppi_times[i]), fontsize=7, ha="left")

    ax.set_xlabel("NCP Build Time (s)")
    ax.set_ylabel("MPC Total Simulation Time (s)")
    ax.set_title("NCP Offline Cost vs MPC Online Cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "build_vs_solve.png", dpi=150)
    plt.close()
    print(f"  Saved build_vs_solve.png")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison figures")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    if args.results_dir:
        base_dir = Path(args.results_dir)
    else:
        base_dir = Path(__file__).parent.parent / "results" / f"phase{args.phase}"

    output_dir = base_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures for phase {args.phase}...")

    results = load_results(base_dir)

    plot_convergence_rates(results, output_dir)
    plot_solve_times(results, output_dir)
    plot_trajectory_cost(results, output_dir)
    plot_ncp_coverage(results, output_dir)
    plot_build_vs_solve(results, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
