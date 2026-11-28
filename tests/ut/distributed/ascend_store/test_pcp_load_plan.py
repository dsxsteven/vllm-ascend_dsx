# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-exact pool restore with discontiguous KV pages and unequal PCP tails."""

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pcp_load_plan import (
    coalesce_copy_descriptors,
    shard_window,
    slice_transfer,
)


@pytest.mark.parametrize("pcp", [2, 3, 8, 16])
@pytest.mark.parametrize("sizes", [[], [0], [1], [511, 0, 513], [131072, 16384, 65], [3, 7, 4097, 0, 17]])
def test_restore_reads_each_source_byte_once(pcp, sizes):
    sizes = np.asarray(sizes, dtype=np.int64)
    sources = [(np.arange(int(n), dtype=np.uint32) * 37 + i * 11).astype(np.uint8) for i, n in enumerate(sizes)]
    # Independent destination pages with guard bytes represent non-contiguous
    # block IDs, cache planes, and a partially filled trailing cache block.
    destinations = [[np.full(int(n) + 32, 0xA5, dtype=np.uint8) for n in sizes] for _ in range(pcp)]
    coverage = [np.zeros(int(n), dtype=np.int16) for n in sizes]
    total = int(sizes.sum())
    plans = [list(shard_window(total, pcp, rank, capacity=1024)) for rank in range(pcp)]
    bytes_per_rank = [0] * pcp
    for window_index in range(len(plans[0])):
        shards = []
        for rank in range(pcp):
            start, end, width, left, right = plans[rank][window_index]
            shard = np.zeros(width, dtype=np.uint8)
            indices, offsets, packed, lengths = slice_transfer(sizes, left, right)
            for i, offset, target, length in zip(indices, offsets, packed, lengths, strict=True):
                shard[target : target + length] = sources[i][offset : offset + length]
                coverage[i][offset : offset + length] += 1
                bytes_per_rank[rank] += int(length)
            shards.append(shard)
        gathered = np.concatenate(shards)
        indices, offsets, packed, lengths = slice_transfer(sizes, start, end)
        for rank in range(pcp):
            for i, offset, source, length in zip(indices, offsets, packed, lengths, strict=True):
                destinations[rank][i][16 + offset : 16 + offset + length] = gathered[source : source + length]
    assert sum(bytes_per_rank) == total
    for i, expected in enumerate(sources):
        np.testing.assert_array_equal(coverage[i], np.ones(len(expected)))
        for rank in range(pcp):
            np.testing.assert_array_equal(destinations[rank][i][16:-16], expected)
            assert np.all(destinations[rank][i][:16] == 0xA5)
            assert np.all(destinations[rank][i][-16:] == 0xA5)


def test_large_layer_uses_one_dynamic_window_and_equal_shards():
    total = (32 * 1024 * 1024 + 1024 * 1024) * 16 + 193
    for rank in range(16):
        windows = list(shard_window(total, 16, rank))
        assert len(windows) == 1
        assert all(width > 32 * 1024 * 1024 for _, _, width, _, _ in windows)
        assert all(width % 512 == 0 for _, _, width, _, _ in windows)
    totals = [sum(right - left for _, _, _, left, right in shard_window(total, 16, r)) for r in range(16)]
    assert sum(totals) == total
    assert max(totals) - min(totals) < 512 * 16


@pytest.mark.parametrize("args", [(-1, 2, 0), (1, 1, 0), (1, 2, 2), (1, 2, -1)])
def test_invalid_topology_rejected(args):
    with pytest.raises(ValueError):
        list(shard_window(*args))


def test_invalid_byte_range_rejected():
    with pytest.raises(ValueError):
        slice_transfer(np.array([7, -1]), 0, 6)
    with pytest.raises(ValueError):
        slice_transfer(np.array([7]), 3, 1)


@pytest.mark.parametrize("position", [0, 1, 5, 10, 15, 20])
def test_empty_interval_has_no_descriptors(position):
    for value in slice_transfer(np.array([10, 10]), position, position):
        assert value.size == 0


@pytest.mark.parametrize("limit", [1, 17, 64, 256])
@pytest.mark.parametrize("seed", [0, 7, 19])
def test_copy_coalescing_preserves_every_byte_and_hole(limit, seed):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 48, size=80, dtype=np.int64)
    src_gaps = rng.choice([0, 0, 0, 9], size=80)
    dst_gaps = rng.choice([0, 0, 0, 13], size=80)
    sources = np.cumsum(lengths + src_gaps) - lengths
    destinations = np.cumsum(lengths + dst_gaps) - lengths
    source = rng.integers(0, 256, size=int(sources[-1] + lengths[-1]), dtype=np.uint8)
    expected = np.full(int(destinations[-1] + lengths[-1]) + 16, 0xA5, dtype=np.uint8)
    actual = expected.copy()
    for src, dst, size in zip(sources, destinations, lengths, strict=True):
        expected[dst : dst + size] = source[src : src + size]
    merged = coalesce_copy_descriptors(sources, destinations, lengths, limit)
    for src, dst, size in zip(*merged, strict=True):
        actual[dst : dst + size] = source[src : src + size]
    np.testing.assert_array_equal(actual, expected)
    assert int(merged[2].sum()) == int(lengths.sum())
    assert np.all((merged[2] > 0) & (merged[2] <= limit))


def test_copy_coalescing_only_merges_both_contiguous_addresses():
    source = np.array([0, 16, 32, 48, 64, 80])
    destination = np.array([100, 116, 200, 216, 216, 232])
    result = coalesce_copy_descriptors(source, destination, np.full(6, 16), 64)
    np.testing.assert_array_equal(result[0], [0, 32, 64])
    np.testing.assert_array_equal(result[1], [100, 200, 216])
    np.testing.assert_array_equal(result[2], [32, 32, 32])
    empty = coalesce_copy_descriptors(np.array([1]), np.array([2]), np.array([0]), 64)
    assert all(x.size == 0 for x in empty)


@pytest.mark.parametrize("case", ["length", "shape", "dimension", "limit"])
def test_invalid_copy_coalescing_is_rejected(case):
    source, destination, length, limit = np.array([0]), np.array([100]), np.array([16]), 64
    if case == "length":
        length[0] = -1
    elif case == "shape":
        source = np.array([0, 16])
    elif case == "dimension":
        source, destination, length = (x.reshape(1, 1) for x in (source, destination, length))
    else:
        limit = 0
    with pytest.raises(ValueError):
        coalesce_copy_descriptors(source, destination, length, limit)
