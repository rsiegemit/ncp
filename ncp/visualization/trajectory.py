"""Trajectory visualization utilities."""

from __future__ import annotations

from typing import Any

import torch


def plot_trajectories(
    trajectories: torch.Tensor,
    *,
    dims: tuple[int, int] = (0, 1),
    ax: Any = None,
    max_lines: int = 50,
    color: str = "green",
    linewidth: float = 0.8,
    alpha: float = 0.6,
    title: str | None = None,
) -> Any:
    """Plot 2-D projections of trajectories.

    Args:
        trajectories: ``(B, T, D)`` tensor of state trajectories.
        dims: Which two state dimensions to project.
        ax: Existing matplotlib Axes.
        max_lines: Maximum number of trajectories to draw.
        color: Line colour.
        linewidth: Line width.
        alpha: Line transparency.
        title: Plot title.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    traj = trajectories.detach().cpu().numpy()
    d0, d1 = dims
    n = min(traj.shape[0], max_lines)

    for i in range(n):
        ax.plot(
            traj[i, :, d0],
            traj[i, :, d1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_xlabel(f"dim {d0}")
    ax.set_ylabel(f"dim {d1}")
    ax.grid(True)
    if title is not None:
        ax.set_title(title)

    return ax
