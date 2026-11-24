# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise window boundaries and partial failures with small CPU staging."""

import ctypes
from functools import partial

import numpy as np
import pytest
import torch

from tests.ut.distributed.ascend_store.test_pcp_load import make_loader
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pcp_load


def window_case(monkeypatch, world, rank):
    capacity = 1024
    monkeypatch.setattr(pcp_load, "SHARD_CAPACITY", capacity)
    monkeypatch.setattr(pcp_load, "shard_window", partial(pcp_load.shard_window, capacity=capacity))
    loader = make_loader(monkeypatch, rank, world)
    lengths = [0, 499, capacity * world + 47, 0, capacity * world + 511]
    sources = [torch.arange(n, dtype=torch.int64).to(torch.uint8) for n in lengths]
    targets = [torch.full((n + 32,), 0xA5, dtype=torch.uint8) for n in lengths]
    gvas = np.array([t.data_ptr() for t in sources], dtype=np.int64)
    addrs = np.array([t.data_ptr() + 16 for t in targets], dtype=np.int64)
    sizes = np.array(lengths, dtype=np.int64)
    state = {"cursor": 0, "gathers": 0, "reads": 0}

    def from_pool(srcs, dsts, counts):
        state["reads"] += 1
        for src, dst, count in zip(srcs, dsts, counts, strict=True):
            ctypes.memmove(int(dst), int(src), int(count))
        return 0

    def gather(output, local, *, group):
        assert group is loader.device_group
        expected = torch.cat(sources)
        cursor = state["cursor"]
        valid = min(output.numel(), expected.numel() - cursor)
        output.zero_()
        output[:valid].copy_(expected[cursor : cursor + valid])
        offset = rank * local.numel()
        output[offset : offset + local.numel()].copy_(local)
        state["cursor"] += valid
        state["gathers"] += 1

    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
    return loader, sources, targets, gvas, addrs, sizes, state, from_pool


def assert_guards_and_bytes(sources, targets):
    for source, target in zip(sources, targets, strict=True):
        assert torch.equal(source, target[16:-16])
        assert torch.all(target[:16] == 0xA5)
        assert torch.all(target[-16:] == 0xA5)


@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (8, 3), (16, 0), (16, 15)])
def test_multiple_windows_restore_and_reuse(monkeypatch, world, rank):
    loader, sources, targets, gvas, addrs, sizes, state, from_pool = window_case(monkeypatch, world, rank)
    for _ in range(2):
        state.update(cursor=0, gathers=0, reads=0)
        for source in sources:
            source.add_(17)
        loader.copy(gvas, addrs, sizes, from_pool)
        assert state["gathers"] == state["reads"] == 3
        assert_guards_and_bytes(sources, targets)
    assert loader.restored_bytes == 2 * int(sizes.sum())


@pytest.mark.parametrize("world", [2, 16])
@pytest.mark.parametrize("remote_failure", [False, True])
def test_later_window_failure_is_not_published_and_can_recover(monkeypatch, world, remote_failure):
    loader, sources, targets, gvas, addrs, sizes, state, from_pool = window_case(monkeypatch, world, world - 1)

    def fail_second_read(*args):
        result = from_pool(*args)
        return -1 if state["reads"] == 2 and not remote_failure else result

    def peer_status(status, **kwargs):
        if remote_failure and state["reads"] == 2:
            status.fill_(1)

    monkeypatch.setattr(pcp_load.dist, "all_reduce", peer_status)
    with pytest.raises(RuntimeError, match="pool read failed"):
        loader.copy(gvas, addrs, sizes, fail_second_read)
    assert state["gathers"] == 1
    assert loader.restored_bytes == 1024 * world
    expected = torch.cat(sources)
    expected[1024 * world :] = 0xA5
    assert torch.equal(torch.cat([t[16:-16] for t in targets]), expected)
    for target in targets:
        assert torch.all(target[:16] == 0xA5) and torch.all(target[-16:] == 0xA5)
    monkeypatch.setattr(pcp_load.dist, "all_reduce", lambda *a, **kw: None)
    state.update(cursor=0, gathers=0, reads=0)
    for source in sources:
        source.add_(29)
    loader.copy(gvas, addrs, sizes, from_pool)
    assert_guards_and_bytes(sources, targets)
    assert loader.restored_bytes == 1024 * world + int(sizes.sum())
