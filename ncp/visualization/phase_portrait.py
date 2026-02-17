"""Phase portrait / cell visualization."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def plot_cells(
    centers: torch.Tensor,
    radii: torch.Tensor,
    *,
    dims: tuple[int, int] = (0, 1),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    pi_ticks: tuple[bool, bool] = (True, False),
    color: str = "blue",
    alpha: float = 0.5,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot a 2-D projection of verified cells as rectangles.

    Args:
        centers: ``(M, D)`` tensor of cell centres.
        radii: ``(M,)`` tensor of half-widths.
        dims: Which two state dimensions to plot.
        xlim: X-axis limits.
        ylim: Y-axis limits.
        pi_ticks: Whether to format ticks as multiples of pi on (x, y).
        color: Rectangle fill colour.
        alpha: Rectangle transparency.
        ax: Existing matplotlib Axes (created if ``None``).
        title: Plot title.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    pts = centers.detach().cpu().numpy()
    rad = radii.detach().cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    d0, d1 = dims
    for i in range(len(pts)):
        x, y = pts[i, d0], pts[i, d1]
        r = rad[i]
        rect = plt.Rectangle(
            (x - r, y - r), 2 * r, 2 * r, color=color, alpha=alpha
        )
        ax.add_patch(rect)

    def _pi_formatter(value: float, _tick: int) -> str:
        n = int(np.round(value / np.pi))
        if n == 0:
            return "0"
        if n == 1:
            return r"$\pi$"
        if n == -1:
            return r"-$\pi$"
        return rf"${n}\pi$"

    if pi_ticks[0]:
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
        ax.xaxis.set_major_formatter(FuncFormatter(_pi_formatter))
    if pi_ticks[1]:
        ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))
        ax.yaxis.set_major_formatter(FuncFormatter(_pi_formatter))

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel(f"dim {d0}")
    ax.set_ylabel(f"dim {d1}")
    ax.grid(True)
    if title is not None:
        ax.set_title(title)

    return ax
