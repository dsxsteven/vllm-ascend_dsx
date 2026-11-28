# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only byte-range planning for PCP-sharded KV pool reads."""

import numpy as np

_COLLECTIVE_ALIGNMENT = 512


def coalesce_copy_descriptors(
    sources: np.ndarray, destinations: np.ndarray, lengths: np.ndarray, max_bytes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join adjacent source AND destination ranges, then bound each copy run.

    Preserve descriptor order, holes and repeated destinations. The source may
    be a packed offset rather than an absolute pointer. Bounding merged runs
    prevents one program from serially copying an entire contiguous KV layer.
    """
    if max_bytes <= 0 or not (sources.shape == destinations.shape == lengths.shape) or lengths.ndim != 1:
        raise ValueError("Invalid copy descriptor shape or run limit")
    if np.any(lengths < 0):
        raise ValueError("Invalid copy descriptor length")
    valid = lengths > 0
    sources, destinations, lengths = (np.asarray(x[valid], dtype=np.int64) for x in (sources, destinations, lengths))
    if not len(lengths):
        return sources, destinations, lengths
    adjacent = (sources[1:] == sources[:-1] + lengths[:-1]) & (destinations[1:] == destinations[:-1] + lengths[:-1])
    starts = np.concatenate(([0], np.flatnonzero(~adjacent) + 1))
    run_lengths = np.add.reduceat(lengths, starts)
    counts = (run_lengths + max_bytes - 1) // max_bytes
    runs = np.repeat(np.arange(len(starts)), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    offsets = (np.arange(len(runs)) - first) * max_bytes
    return (
        sources[starts[runs]] + offsets,
        destinations[starts[runs]] + offsets,
        np.minimum(max_bytes, run_lengths[runs] - offsets),
    )


def slice_transfer(sizes: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map packed bytes to descriptor indices, source/packed offsets and sizes."""
    if start < 0 or end < start or np.any(sizes < 0):
        raise ValueError("Invalid packed KV byte range")
    ends = np.cumsum(sizes, dtype=np.int64)
    return slice_transfer_bounds(ends - sizes, ends, start, end)


def slice_transfer_bounds(
    starts: np.ndarray, ends: np.ndarray, start: int, end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice a validated nonnegative layout whose cumulative bounds are reused."""
    first = int(np.searchsorted(ends, start, side="right"))
    last = int(np.searchsorted(starts, end, side="left"))
    indices = np.arange(first, max(first, last), dtype=np.intp)
    indices = indices[(ends[indices] > starts[indices]) & (end > start)]
    left = np.maximum(starts[indices], start)
    right = np.minimum(ends[indices], end)
    return indices, left - starts[indices], left - start, right - left


def shard_window(total_bytes: int, world_size: int, rank: int, capacity: int | None = None):
    """Yield window bounds, equal collective bytes, and the rank's byte range."""
    if total_bytes < 0 or world_size < 2 or not 0 <= rank < world_size:
        raise ValueError("Invalid PCP load topology or byte count")
    if capacity is None:
        capacity = max(
            _COLLECTIVE_ALIGNMENT,
            ((total_bytes + world_size * _COLLECTIVE_ALIGNMENT - 1) // (world_size * _COLLECTIVE_ALIGNMENT))
            * _COLLECTIVE_ALIGNMENT,
        )
    if capacity <= 0 or capacity % _COLLECTIVE_ALIGNMENT:
        raise ValueError("PCP shard capacity must be a positive multiple of 512 bytes")
    for start in range(0, total_bytes, capacity * world_size):
        end = min(start + capacity * world_size, total_bytes)
        shard_bytes = (end - start + world_size - 1) // world_size
        shard_bytes = (shard_bytes + _COLLECTIVE_ALIGNMENT - 1) // _COLLECTIVE_ALIGNMENT * _COLLECTIVE_ALIGNMENT
        local_start = min(start + rank * shard_bytes, end)
        local_end = min(start + (rank + 1) * shard_bytes, end)
        yield start, end, shard_bytes, local_start, local_end
