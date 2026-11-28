# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred CPU streams check actual gather/scatter/data-copy dependencies."""

from contextlib import contextmanager

import pytest
import torch

from tests.ut.distributed.ascend_store.test_pcp_load import byte_case, install_gather, make_loader, read_bytes
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pcp_load


class DeferredStream:
    def __init__(self):
        self.operations = []
        self.completed = 0
        self.running = False
        self.host_waits = 0

    def submit(self, operation):
        self.operations.append(operation)

    def drain(self, end=None):
        end = len(self.operations) if end is None else end
        if self.completed >= end:
            return
        assert not self.running, "cross-stream dependency cycle"
        self.running = True
        try:
            while self.completed < end:
                operation = self.operations[self.completed]
                operation()
                self.completed += 1
        finally:
            self.running = False

    def wait_event(self, event):
        source, end = event.fence  # Snapshot this recording, not a later reuse.
        self.submit(lambda: source.drain(end))

    def synchronize(self):
        self.host_waits += 1
        self.drain()


class DeferredEvent:
    def record(self, stream):
        self.fence = stream, len(stream.operations)


@pytest.mark.parametrize("family", ["A3", "A5"])
@pytest.mark.parametrize("world,rank", [(2, 0), (8, 7), (16, 15)])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("peer_failure", [False, True])
def test_single_gather_deferred_stream_lifetimes(monkeypatch, family, world, rank, prefetch, peer_failure):
    loader = make_loader(monkeypatch, rank, world, family, True, True)
    loader._ensure_capacity(16384)
    loader.stream, loader.scatter_stream = DeferredStream(), DeferredStream()
    loader._gather_ready, loader._batch_comm_done = DeferredEvent(), DeferredEvent()
    active = [loader.stream]
    calls = []

    @contextmanager
    def stream_context(stream):
        old = active[0]
        active[0] = stream
        try:
            yield
        finally:
            active[0] = old

    monkeypatch.setattr(torch.npu, "stream", stream_context)
    original_copy = torch.Tensor.copy_

    def copy_tensor(target, source, *args, **kwargs):
        metadata = loader._device_descriptors
        if (
            metadata is not None and target.untyped_storage().data_ptr() == metadata.untyped_storage().data_ptr()
        ) or target is loader._host_result:
            active[0].submit(lambda: original_copy(target, source, *args, **kwargs))
            return target
        return original_copy(target, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_tensor)
    original_scatter = pcp_load.checked_pcp_scatter

    def scatter(*args):
        calls.append("scatter_submit")
        active[0].submit(lambda: original_scatter(*args))

    monkeypatch.setattr(pcp_load, "checked_pcp_scatter", scatter)
    for lengths in [[101, 0, 8193], [1], [4097, 29]]:
        sources, targets, gvas, addrs, sizes = byte_case(loader, lengths)
        install_gather(monkeypatch, loader, sources, 0 if peer_failure else None)
        original_gather = pcp_load.dist.all_gather_into_tensor

        def gather(*args, _gather=original_gather, **kwargs):
            calls.append("gather_submit")
            active[0].submit(lambda: _gather(*args, **kwargs))

        monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
        token = object()
        if prefetch:
            loader.prefetch(gvas, sizes, read_bytes, token)
        if peer_failure:
            with pytest.raises(RuntimeError, match="pool read failed"):
                loader.copy(gvas, addrs, sizes, read_bytes, token=token)
            assert all(torch.all(t == 0xA5) for t in targets)
        else:
            loader.copy(gvas, addrs, sizes, read_bytes, token=token)
            assert all(torch.equal(s, t[16:-16]) for s, t in zip(sources, targets, strict=True))
        assert loader.stream.completed == len(loader.stream.operations)
        assert loader.scatter_stream.completed == len(loader.scatter_stream.operations)
    assert calls == ["gather_submit", "scatter_submit"] * 3
    assert loader.stream.host_waits == 0
    assert loader.scatter_stream.host_waits == 3
