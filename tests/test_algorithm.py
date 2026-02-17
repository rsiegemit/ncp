"""Tests for algorithm modules."""

import torch
import pytest

from ncp.algorithm.search import search_alpha_parallel, certify_alpha
from ncp.algorithm.result import NCPResult
from ncp.utils.wrapping import wrap_angles
from ncp.utils.tensor_ops import pad_tensor_rows_1d


class TestSearchAlpha:
    def test_shape(self) -> None:
        B = 5
        T = 10
        sol = torch.rand(B, T + 1)
        v_start = sol[:, 0] + 0.1  # ensure V_start > radius
        radius = torch.full((B,), 0.01)
        t_eval = torch.linspace(0, 1, T + 1)
        alpha, indices = search_alpha_parallel(sol, v_start, radius, t_eval, 1.0)
        assert alpha.shape == (B,)
        assert indices.shape == (B,)

    def test_positive_alpha_for_contracting(self) -> None:
        # Construct a trajectory that clearly contracts
        T = 20
        t_eval = torch.linspace(0, 2, T + 1)
        sol = torch.exp(-t_eval).unsqueeze(0)  # exponential decay
        v_start = torch.tensor([1.0])
        radius = torch.tensor([0.001])
        alpha, _ = search_alpha_parallel(sol, v_start, radius, t_eval, 0.5)
        assert alpha.item() > 0


class TestCertifyAlpha:
    def test_certifies_contracting(self) -> None:
        T = 20
        t_eval = torch.linspace(0, 2, T + 1)
        sol = torch.exp(-2 * t_eval).unsqueeze(0)
        v_start = torch.tensor([1.0])
        radius = torch.tensor([0.001])
        cond, idx = certify_alpha(sol, v_start, radius, t_eval, 0.5, 0.01)
        assert cond.item() is True


class TestWrapAngles:
    def test_basic(self) -> None:
        x = torch.tensor([[4.0, 1.0]])
        wrapped = wrap_angles(x, dims=(0,))
        assert -torch.pi < wrapped[0, 0].item() <= torch.pi
        assert wrapped[0, 1].item() == 1.0

    def test_no_wrap(self) -> None:
        x = torch.tensor([[4.0, 5.0, 6.0]])
        wrapped = wrap_angles(x, dims=())
        assert torch.equal(wrapped, x)


class TestPadTensor:
    def test_basic(self) -> None:
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx = torch.tensor([2, 1])
        result = pad_tensor_rows_1d(t, idx, sentinel=0.0)
        expected = torch.tensor([[1.0, 2.0, 0.0], [4.0, 0.0, 0.0]])
        assert torch.equal(result, expected)


class TestNCPResult:
    def test_dataclass(self) -> None:
        r = NCPResult(
            verified_centers=torch.zeros(1, 2),
            verified_radii=torch.tensor([0.1]),
            verified_controls=torch.zeros(1, 5),
            verified_alphas=torch.tensor([0.05]),
            verified_indices=torch.tensor([1]),
        )
        assert r.verified_centers.shape == (1, 2)
        assert r.unverified_centers is None
