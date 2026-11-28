# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-exact KV scatter guarded by the statuses in the same AllGather."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["count", "payload_bytes"])
def _checked_scatter(
    src_ptrs,
    dst_ptrs,
    sizes,
    gathered,
    result,
    count,
    payload_bytes,
    WORLD: tl.constexpr,
    STATUS_BYTES: tl.constexpr,
    RANK_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    ranks = tl.arange(0, RANK_BLOCK).to(tl.int64)
    # Every program checks all ranks before any destination write. No program
    # depends on another program's result, so failure cannot race publication.
    flags = tl.load(gathered + ranks * (payload_bytes + STATUS_BYTES) + payload_bytes, mask=ranks < WORLD, other=0)
    failed = tl.max(flags.to(tl.int32), 0) != 0
    if pid == 0:
        tl.store(result, failed.to(tl.uint8))
    if (pid < count) & ~failed:
        src = tl.load(src_ptrs + pid).to(tl.pointer_type(tl.uint8))
        dst = tl.load(dst_ptrs + pid).to(tl.pointer_type(tl.uint8))
        size = tl.load(sizes + pid)
        offsets = tl.arange(0, BLOCK_SIZE)
        for start in range(0, size, BLOCK_SIZE):
            mask = start + offsets < size
            values = tl.load(src + start + offsets, mask=mask, other=0, cache_modifier=".cg")
            tl.store(dst + start + offsets, values, mask=mask, cache_modifier=".cg")


def checked_pcp_scatter(pointers, gathered, result, payload_bytes, world_size, status_bytes, block_size):
    """Write bytes only if every rank trailer is zero; always write result[0].

    ``pointers`` is a contiguous int64 (3, N) tensor of source pointers,
    destination pointers and lengths. AllGather is complete on this stream.
    ``gathered`` and ``result`` use uint8. The caller retains all buffers until
    stream completion. Payload length is runtime data, not a constexpr.
    """
    if pointers.ndim != 2 or pointers.shape[0] != 3 or not pointers.is_contiguous():
        raise ValueError("PCP scatter requires contiguous (3, N) descriptors")
    if pointers.dtype != torch.int64 or gathered.dtype != torch.uint8 or result.dtype != torch.uint8:
        raise ValueError("PCP scatter requires int64 descriptors and uint8 data/status")
    if not gathered.is_contiguous() or not result.is_contiguous():
        raise ValueError("PCP gather/result buffers must be contiguous")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("PCP scatter block size must be a positive power of two")
    if payload_bytes < 0 or world_size < 2 or status_bytes <= 0:
        raise ValueError("Invalid PCP status layout")
    if gathered.numel() < (payload_bytes + status_bytes) * world_size or result.numel() != 1:
        raise ValueError("PCP gather/result buffers are too small")
    count = pointers.shape[1]
    _checked_scatter[(max(1, count),)](
        pointers[0],
        pointers[1],
        pointers[2],
        gathered,
        result,
        count,
        payload_bytes,
        WORLD=world_size,
        STATUS_BYTES=status_bytes,
        RANK_BLOCK=triton.next_power_of_2(world_size),
        BLOCK_SIZE=block_size,
    )
