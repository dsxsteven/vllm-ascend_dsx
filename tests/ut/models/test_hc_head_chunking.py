# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for hc_head token-chunking correctness."""

import pytest
import torch


class _FakeHcNorm:
    """RMSNorm without weight, fp32 — mirrors DeepseekV4Model.hc_norm."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        return x * rsqrt


def _reference_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    x_norm = hc_norm(x)
    mixes = torch.nn.functional.linear(x_norm, hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def _chunked_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm, chunk):
    shape, dtype = x.size(), x.dtype
    num_tokens = shape[0]
    c = min(chunk, num_tokens)
    y = torch.empty(num_tokens, shape[2], dtype=torch.float32, device=x.device)
    for start in range(0, num_tokens, c):
        end = min(start + c, num_tokens)
        xc = x[start:end]
        xc_shape = xc.size()
        xc_flat = xc.flatten(1).float()
        xc_norm = hc_norm(xc_flat)
        mixes = torch.nn.functional.linear(xc_norm, hc_fn)
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + eps
        y[start:end] = torch.sum(pre.unsqueeze(-1) * xc_flat.view(xc_shape), dim=1)
    return y.to(dtype)


class TestHcHeadChunking:
    def _make(self, N, hc_mult=4, H=256, seed=42):
        torch.manual_seed(seed)
        hc_dim = hc_mult * H
        x = torch.randn(N, hc_mult, H, dtype=torch.bfloat16)
        hc_fn = torch.randn(hc_mult, hc_dim, dtype=torch.float32) * 0.02
        hc_scale = torch.randn(1, dtype=torch.float32) * 0.1
        hc_base = torch.randn(hc_mult, dtype=torch.float32) * 0.1
        hc_norm = _FakeHcNorm()
        eps = 1e-6
        return x, hc_fn, hc_scale, hc_base, eps, hc_norm

    @pytest.mark.parametrize("N,chunk", [
        (1, 4096),
        (100, 4096),
        (4097, 4096),
        (8192, 4096),
        (22000, 4096),
        (22000, 1024),
        (22000, 512),
    ])
    def test_chunking_matches_reference(self, N, chunk):
        x, hc_fn, hc_scale, hc_base, eps, hc_norm = self._make(N)
        ref = _reference_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm)
        out = _chunked_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm, chunk)
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-4)

    def test_output_dtype_preserved(self):
        x, hc_fn, hc_scale, hc_base, eps, hc_norm = self._make(100)
        out = _chunked_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm, 32)
        assert out.dtype == x.dtype

    def test_output_shape(self):
        x, hc_fn, hc_scale, hc_base, eps, hc_norm = self._make(100, H=512)
        out = _chunked_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm, 32)
        assert out.shape == (100, 512)

    def test_chunk_boundary_exact(self):
        """chunk boundary at exact multiple: no off-by-one."""
        x, hc_fn, hc_scale, hc_base, eps, hc_norm = self._make(8192)
        ref = _reference_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm)
        out = _chunked_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm, 4096)
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-4)

    def test_remainder_chunk(self):
        """N not divisible by chunk: last partial chunk handled correctly."""
        x, hc_fn, hc_scale, hc_base, eps, hc_norm = self._make(4097)
        ref = _reference_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm)
        out = _chunked_hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_norm, 4096)
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-4)
