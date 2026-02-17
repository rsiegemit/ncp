"""Tensor utility operations."""

from __future__ import annotations

import torch


def pad_tensor_rows_1d(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    sentinel: float = 1337.0,
) -> torch.Tensor:
    """Replace trailing columns in each row with a sentinel value.

    For row *i* only the first ``indices[i]`` columns are kept;
    the rest are filled with *sentinel*.

    Args:
        tensor: Input tensor of shape ``(m, n)``.
        indices: 1-D integer tensor of shape ``(m,)`` giving the number
            of valid columns per row.
        sentinel: Value used for padding.

    Returns:
        A new tensor with sentinel padding applied.
    """
    m, n = tensor.shape
    column_indices = torch.arange(n, device=tensor.device).expand(m, n)
    mask = column_indices < indices.unsqueeze(1)
    return tensor.masked_fill(~mask, sentinel)
