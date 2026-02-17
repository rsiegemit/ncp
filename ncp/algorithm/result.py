"""Result dataclass for NCP computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NCPResult:
    """Stores the output of an NCP computation.

    Attributes:
        verified_centers: ``(M, D)`` centres of verified cells.
        verified_radii: ``(M,)`` half-widths of verified cells.
        verified_controls: ``(M, T, [ctrl_dim])`` control sequences.
        verified_alphas: ``(M,)`` contraction rates.
        verified_indices: ``(M,)`` minimising time indices.
        unverified_centers: Optional ``(N, D)`` centres of unverified cells.
        unverified_radii: Optional ``(N,)`` half-widths of unverified cells.
        elapsed_seconds: Wall-clock time of the computation.
    """

    verified_centers: torch.Tensor
    verified_radii: torch.Tensor
    verified_controls: torch.Tensor
    verified_alphas: torch.Tensor
    verified_indices: torch.Tensor
    unverified_centers: torch.Tensor | None = None
    unverified_radii: torch.Tensor | None = None
    elapsed_seconds: float = 0.0
