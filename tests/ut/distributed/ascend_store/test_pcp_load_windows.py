# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank payload boundaries must skip in-band trailers without losing KV bytes."""

import pytest
import torch

from tests.ut.distributed.ascend_store.test_pcp_load import byte_case, install_gather, make_loader, read_bytes


@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (8, 3), (16, 0), (16, 15)])
def test_rank_boundaries_restore_and_reuse(monkeypatch, world, rank):
    loader = make_loader(monkeypatch, rank, world)
    sources, targets, gvas, addrs, sizes = byte_case(loader, [0, 499, 1024 * world + 47, 0, 1024 * world + 511])
    calls = install_gather(monkeypatch, loader, sources)
    for _ in range(2):
        for source in sources:
            source.add_(17)
        loader.copy(gvas, addrs, sizes, read_bytes)
        for source, target in zip(sources, targets, strict=True):
            assert torch.equal(source, target[16:-16])
            assert torch.all(target[:16] == 0xA5) and torch.all(target[-16:] == 0xA5)
    assert len(calls) == 2
    assert loader.restored_bytes == 2 * int(sizes.sum())


@pytest.mark.parametrize("world", [2, 16])
@pytest.mark.parametrize("remote_failure", [False, True])
def test_partial_read_failure_rejects_entire_layer(monkeypatch, world, remote_failure):
    loader = make_loader(monkeypatch, world - 1, world)
    sources, targets, gvas, addrs, sizes = byte_case(loader, [0, 499, 1024 * world + 47, 0, 1024 * world + 511])
    calls = install_gather(monkeypatch, loader, sources, 0 if remote_failure else None)

    def partial_failure(srcs, dsts, lengths):
        # The backend may already have copied some ranges when a later range fails.
        read_bytes(srcs[:1], dsts[:1], lengths[:1])
        return 0 if remote_failure else -1

    with pytest.raises(RuntimeError, match="pool read failed"):
        loader.copy(gvas, addrs, sizes, partial_failure)
    assert len(calls) == 1 and loader.restored_bytes == 0
    assert all(torch.all(target == 0xA5) for target in targets)
    install_gather(monkeypatch, loader, sources)
    loader.copy(gvas, addrs, sizes, read_bytes)
    assert all(torch.equal(source, target[16:-16]) for source, target in zip(sources, targets, strict=True))
