"""Angle wrapping utilities for periodic state dimensions."""

from __future__ import annotations

import math

import torch

_TWO_PI = 2 * math.pi


def wrap_angles(
    tensor: torch.Tensor,
    dims: tuple[int, ...] = (0,),
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Wrap specified dimensions of a tensor to (-pi, pi].

    Works on any tensor whose last axis is the state dimension.
    For example, a trajectory tensor of shape ``(B, T, D)`` with
    ``dims=(0,)`` wraps index 0 of the last axis.

    Args:
        tensor: Input tensor with state dimension along the last axis.
        dims: Indices (into the last axis) that should be wrapped.
        inplace: If True, modify the tensor in place (caller must own it).

    Returns:
        A tensor with the specified dimensions wrapped.
    """
    if not dims:
        return tensor
    result = tensor if inplace else tensor.clone()
    for d in dims:
        result[..., d] = (result[..., d] + math.pi) % _TWO_PI - math.pi
    return result
