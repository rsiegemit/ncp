"""Lookup-based controller from NCP results."""

from __future__ import annotations

import torch

from ncp.algorithm.result import NCPResult
from ncp.geometry.hypercubes import find_hypercubes
from ncp.utils.tensor_ops import pad_tensor_rows_1d


class NCPController:
    """A look-up controller that maps states to control sequences.

    Given an :class:`NCPResult`, this controller finds the containing
    verified cell and returns its stored control sequence.

    Args:
        result: Output of :meth:`NCPBuilder.build`.
        sentinel: Value used to mark inactive time steps.
    """

    def __init__(
        self,
        result: NCPResult,
        sentinel: float = 1337.0,
    ) -> None:
        self.centers = result.verified_centers
        self.radii = result.verified_radii
        self.controls = result.verified_controls
        self.indices = result.verified_indices
        self.alphas = result.verified_alphas
        self.sentinel = sentinel

    @torch.no_grad()
    def lookup(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up control sequences for the given states.

        Args:
            states: ``(B, D)`` tensor of query states.

        Returns:
            ``(controls, taus)`` — the sentinel-padded control sequences
            and the number of valid time steps per state.
            If a state does not fall in any verified cell, its control
            is filled entirely with sentinels and its tau is 0.
        """
        idx = find_hypercubes(states, self.centers, self.radii)
        found = idx > -1

        # Default: sentinel-filled controls
        if self.controls.dim() == 2:
            out_ctrl = torch.full_like(
                self.controls[:1].expand(states.shape[0], -1), self.sentinel
            )
        else:
            out_ctrl = torch.full_like(
                self.controls[:1].expand(states.shape[0], -1, -1), self.sentinel
            )
        out_taus = torch.zeros(states.shape[0], dtype=torch.long, device=states.device)

        if found.any():
            matched_idx = idx[found]
            matched_ctrl = self.controls[matched_idx]
            matched_times = self.indices[matched_idx].long()

            if matched_ctrl.dim() == 2:
                matched_ctrl = pad_tensor_rows_1d(
                    matched_ctrl, matched_times, self.sentinel
                )

            out_ctrl[found] = matched_ctrl
            out_taus[found] = matched_times

        return out_ctrl, out_taus
