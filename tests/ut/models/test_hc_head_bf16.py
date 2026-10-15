# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for hc_head bf16 memory optimization correctness."""

import pytest
import torch


def _reference_hc_head(x, hc_fn, hc_scale, hc_base, eps):
    """Original fp32 full-batch implementation (ground truth)."""
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    x_norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    mixes = torch.nn.functional.linear(x_norm, hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def _bf16_hc_head(x, hc_fn, hc_scale, hc_base, eps, chunk):
    """Optimized bf16 implementation (mirrors model.py after optimization B)."""
    shape, dtype = x.size(), x.dtype
    num_tokens = shape[0]
    c = min(chunk, num_tokens)
    # Cast the tiny weight (hc_mult × hc_dim × 4B ≈ 256 KB) to input dtype
    # so F.linear doesn't need a fp32 copy of the large activation.
    hc_fn_cast = hc_fn.to(dtype)
    y = torch.empty(num_tokens, shape[2], dtype=dtype, device=x.device)
    for start in range(0, num_tokens, c):
        end = min(start + c, num_tokens)
        xc = x[start:end]
        xc_shape = xc.size()
        xc_flat = xc.flatten(1)
        inv_norm = torch.rsqrt(
            xc_flat.float().square().mean(-1, keepdim=True) + 1e-6
        )
        mixes = torch.nn.functional.linear(xc_flat * inv_norm.to(dtype), hc_fn_cast)
        pre = torch.sigmoid(mixes.float() * hc_scale + hc_base) + eps
        y[start:end] = torch.sum(
            pre.to(dtype).unsqueeze(-1) * xc_flat.view(xc_shape), dim=1
        )
    return y


class TestHcHeadBf16:
    def _make(self, N, hc_mult=4, H=256, seed=42):
        torch.manual_seed(seed)
        hc_dim = hc_mult * H
        x = torch.randn(N, hc_mult, H, dtype=torch.bfloat16)
        hc_fn = torch.randn(hc_mult, hc_dim, dtype=torch.float32) * 0.02
        hc_scale = torch.randn(1, dtype=torch.float32) * 0.1
        hc_base = torch.randn(hc_mult, dtype=torch.float32) * 0.1
        return x, hc_fn, hc_scale, hc_base, 1e-6

    @pytest.mark.parametrize("N,chunk", [
        (1, 4096), (100, 4096), (2750, 4096), (2750, 1024),
        (4096, 4096), (4097, 4096), (8192, 4096), (22000, 4096),
    ])
    def test_bf16_matches_reference(self, N, chunk):
        """bf16 output must be close to fp32 reference within bf16 tolerance.

        bf16 has ~3 decimal digits (8-bit mantissa); intermediate bf16 ops
        (norm scaling, gating product) compound to at most ~2 bf16 ulps.
        For values in the [-1, 1] range, that's ≈ 0.016 absolute.
        Near-zero values can have large relative error but tiny absolute.
        """
        x, hc_fn, hc_scale, hc_base, eps = self._make(N)
        ref = _reference_hc_head(x, hc_fn, hc_scale, hc_base, eps)
        out = _bf16_hc_head(x, hc_fn, hc_scale, hc_base, eps, chunk)
        # Check absolute error is within ~2 bf16 ulps of the reference magnitude
        ref_abs = ref.float().abs()
        # bf16 ulp at |x| ≈ 0.5-1.0 is 2^-8 ≈ 0.004; allow 4 ulps ≈ 0.016
        tolerance = torch.clamp(ref_abs * 0.04, min=0.002, max=0.05)
        diff = (out.float() - ref.float()).abs()
        # Allow at most 8% of elements to exceed tolerance (bf16 edge values)
        exceed = (diff > tolerance).float().mean()
        assert exceed < 0.08, (
            f"bf16 mismatch: {exceed:.1%} elements exceed tolerance, "
            f"max_diff={diff.max():.6f}, max_ref={ref_abs.max():.4f}"
        )

    def test_output_dtype_bf16(self):
        x, hc_fn, hc_scale, hc_base, eps = self._make(100)
        out = _bf16_hc_head(x, hc_fn, hc_scale, hc_base, eps, 32)
        assert out.dtype == torch.bfloat16

    def test_output_shape(self):
        x, hc_fn, hc_scale, hc_base, eps = self._make(100, H=512)
        out = _bf16_hc_head(x, hc_fn, hc_scale, hc_base, eps, 32)
        assert out.shape == (100, 512)

    def test_all_zero_input(self):
        """Edge case: zero input should produce zero output (sum of zero)."""
        x, hc_fn, hc_scale, hc_base, eps = self._make(100)
        x = torch.zeros_like(x)
        ref = _reference_hc_head(x, hc_fn, hc_scale, hc_base, eps)
        out = _bf16_hc_head(x, hc_fn, hc_scale, hc_base, eps, 64)
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-3)

    def test_large_values_stable(self):
        """Edge case: large inputs shouldn't overflow bf16."""
        x, hc_fn, hc_scale, hc_base, eps = self._make(100)
        x = x * 10.0
        ref = _reference_hc_head(x, hc_fn, hc_scale, hc_base, eps)
        out = _bf16_hc_head(x, hc_fn, hc_scale, hc_base, eps, 64)
        # Just check no NaN/Inf and same order of magnitude
        assert torch.isfinite(out.float()).all()
        assert (out.float().abs().max() / ref.float().abs().max()) < 1.1
        assert (out.float().abs().max() / ref.float().abs().max()) > 0.9
