# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token slices for the canonical and rank-local PCP prefill layouts.

PCP's DualChunkSwap represents disjoint portions of one request as separate
local rows. Work from those rows, never infer a contiguous global rank slice.
All geometry is computed from CPU mirrors; device sequence lengths are sliced
with the same offsets without synchronizing them back to the host.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PCPChunkGeometry:
    start: int
    end: int
    request_indices: torch.Tensor
    query_start_loc: torch.Tensor
    trailing_tokens: torch.Tensor


@dataclass(frozen=True)
class DSAPCPChunk:
    start: int
    end: int
    metadata: Any


def compute_pcp_chunk_geometries(
    query_start_loc_cpu: torch.Tensor,
    num_actual_tokens: int,
    chunk_size: int,
) -> tuple[PCPChunkGeometry, ...]:
    """Partition real rows, dropping zero-token requests and padding."""
    if chunk_size <= 0:
        raise ValueError("PCP attention chunk_size must be positive")
    if query_start_loc_cpu.device.type != "cpu":
        raise ValueError("PCP chunk geometry requires CPU query offsets")
    offsets = query_start_loc_cpu.to(dtype=torch.int64)
    if offsets.ndim != 1 or offsets.numel() < 1 or int(offsets[0]) != 0:
        raise ValueError("PCP query offsets must be a vector beginning at zero")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("PCP query offsets must be nondecreasing")
    if num_actual_tokens < 0 or int(offsets[-1]) < num_actual_tokens:
        raise ValueError("PCP query offsets must cover all actual tokens")
    chunks = []
    for start in range(0, num_actual_tokens, chunk_size):
        end = min(start + chunk_size, num_actual_tokens)
        left = offsets[:-1].clamp(min=start, max=end)
        right = offsets[1:].clamp(min=start, max=end)
        indices = torch.nonzero(right > left, as_tuple=True)[0]
        lengths = (right - left)[indices]
        query_start_loc = torch.cat((offsets.new_zeros(1), lengths.cumsum(0)))
        chunks.append(
            PCPChunkGeometry(
                start=start,
                end=end,
                request_indices=indices,
                query_start_loc=query_start_loc.to(query_start_loc_cpu.dtype),
                trailing_tokens=offsets[1:][indices] - right[indices],
            )
        )
    return tuple(chunks)


def slice_pcp_chunk_common_metadata(common: Any, geometry: PCPChunkGeometry) -> Any:
    """Preserve absolute causal endpoints and request-specific cache tables."""
    indices = geometry.request_indices

    def select_rows(value):
        return value.index_select(0, indices.to(value.device)) if value is not None else None

    seq_lens_cpu = common._seq_lens_cpu
    if seq_lens_cpu is None:
        seq_lens_cpu = common.seq_lens_cpu
    if seq_lens_cpu is None or seq_lens_cpu.device.type != "cpu":
        raise ValueError("PCP prefill chunking requires exact CPU sequence lengths")
    trailing = geometry.trailing_tokens
    chunk_seq_cpu = select_rows(seq_lens_cpu) - trailing.to(seq_lens_cpu.dtype)
    chunk_seq = select_rows(common.seq_lens) - trailing.to(common.seq_lens.device, common.seq_lens.dtype)
    qsl_cpu = geometry.query_start_loc
    query_lens = qsl_cpu[1:] - qsl_cpu[:-1]
    computed_cpu = chunk_seq_cpu - query_lens
    if bool((computed_cpu < 0).any()):
        raise ValueError("PCP chunk query length exceeds its sequence length")
    start, end = geometry.start, geometry.end
    upper_bound = select_rows(common.seq_lens_cpu_upper_bound)
    if upper_bound is not None:
        upper_bound = upper_bound - trailing.to(upper_bound.device, upper_bound.dtype)
    # Construct a new common object rather than retain stale per-request
    # mirrors, token-to-request maps, graph padding or unrelated CP metadata.
    return type(common)(
        query_start_loc=qsl_cpu.to(common.query_start_loc.device),
        query_start_loc_cpu=qsl_cpu,
        seq_lens=chunk_seq,
        seq_lens_cpu=chunk_seq_cpu,
        _seq_lens_cpu=chunk_seq_cpu,
        num_computed_tokens_cpu=computed_cpu,
        _num_computed_tokens_cpu=computed_cpu,
        seq_lens_cpu_upper_bound=upper_bound,
        num_reqs=indices.numel(),
        num_actual_tokens=end - start,
        num_input_tokens=end - start,
        max_query_len=int(query_lens.max()),
        max_seq_len=int(chunk_seq_cpu.max()),
        block_table_tensor=select_rows(common.block_table_tensor),
        slot_mapping=common.slot_mapping[start:end],
        positions=common.positions[start:end],
        positions_cpu=common.positions_cpu[start:end] if common.positions_cpu is not None else None,
        causal=common.causal,
        attn_state=common.attn_state,
        is_prefilling=select_rows(common.is_prefilling),
    )
