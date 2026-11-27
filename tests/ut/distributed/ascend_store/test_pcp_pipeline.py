# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU byte checks and deferred-stream simulation for PCP restore pipelining."""

import ctypes
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.ut.distributed.ascend_store.test_pcp_load import make_loader
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pcp_load, pcp_load_plan, pool_worker


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


def defer_metadata_copy(monkeypatch, loader, active, calls):
    original_copy = torch.Tensor.copy_

    def copy_tensor(target, source, *args, **kwargs):
        device = loader._device_descriptors
        if device is not None and target.untyped_storage().data_ptr() == device.untyped_storage().data_ptr():
            assert target.is_contiguous() and source.is_contiguous(), "metadata DMA must not need device packing"
            calls.append("metadata_submit")
            # Retain the pinned host view to detect premature host-buffer reuse.
            active[0].submit(lambda: original_copy(target, source, *args, **kwargs))
            return target
        return original_copy(target, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_tensor)
    return original_copy


def pipeline_case(monkeypatch, family, world, rank):
    capacity = 512
    monkeypatch.setattr(pcp_load, "SHARD_CAPACITY", capacity)
    monkeypatch.setattr(
        pcp_load, "shard_window", lambda total, size, local: pcp_load_plan.shard_window(total, size, local, capacity)
    )
    loader = make_loader(monkeypatch, rank, world)
    loader.prefetch_enabled = loader.overlap_enabled = True
    loader.batch_windows = 3
    loader.locals = [torch.empty(capacity, dtype=torch.uint8) for _ in range(3)]
    loader.local = loader.locals[0]
    loader.gathered_buffers = [torch.empty(capacity * world, dtype=torch.uint8) for _ in range(2)]
    loader.gathered = loader.gathered_buffers[0]
    loader.stream, loader.scatter_stream = DeferredStream(), DeferredStream()
    loader._gather_ready = [DeferredEvent(), DeferredEvent()]
    loader._scatter_done = [DeferredEvent(), DeferredEvent()]
    loader._batch_comm_done = DeferredEvent()
    profile = get_hardware_profile(getattr(AscendDeviceType, family))
    loader.copy_block_bytes = profile.pcp_kv_scatter_block_bytes
    loader._read_status = torch.empty((), dtype=torch.int32) if profile.pcp_kv_device_status_reduce else None
    active = [loader.stream]
    calls = []

    @contextmanager
    def stream_context(stream):
        previous = active[0]
        active[0] = stream
        try:
            yield
        finally:
            active[0] = previous

    monkeypatch.setattr(torch.npu, "stream", stream_context)
    original_copy = defer_metadata_copy(monkeypatch, loader, active, calls)
    sizes = np.array([101, 0, world * capacity * 6 - 101, 7], dtype=np.int64)
    sources = [torch.arange(int(n), dtype=torch.int64).to(torch.uint8) for n in sizes]
    targets = [torch.full((int(n) + 32,), 0xA5, dtype=torch.uint8) for n in sizes]
    gvas = np.array([x.data_ptr() for x in sources], dtype=np.int64)
    addrs = np.array([x.data_ptr() + 16 for x in targets], dtype=np.int64)
    plan = loader._get_transfer_plan(sizes)
    state = dict(gathers=0, scatters=0)

    def read(srcs, dsts, lengths):
        assert loader.stream.completed == len(loader.stream.operations), "local shard reused before gather completion"
        for src, dst, length in zip(srcs, dsts, lengths, strict=True):
            ctypes.memmove(int(dst), int(src), int(length))
        return 0

    def gather(output, local, *, group):
        assert group is loader.device_group and active[0] is loader.stream
        index = state["gathers"]
        state["gathers"] += 1
        window = plan[index % len(plan)]

        def execute():
            calls.append(("gather", index))
            start, end, width, valid, *_ = window
            expected = torch.cat(sources)[start:end]
            offset = rank * width
            assert torch.equal(local[:valid], expected[offset : offset + valid])
            assert torch.all(local[valid:] == 0)
            output.zero_()
            original_copy(output[: end - start], expected)

        active[0].submit(execute)

    def scatter(srcs, dsts, lengths, **kwargs):
        assert active[0] is loader.scatter_stream
        index = state["scatters"]
        state["scatters"] += 1

        def execute():
            calls.append(("scatter", index))
            for src, dst, length in zip(srcs.tolist(), dsts.tolist(), lengths.tolist(), strict=True):
                ctypes.memmove(dst, src, length)

        active[0].submit(execute)

    def finish_copy_stream():
        loader.scatter_stream.host_waits += 1
        # Exercise a legal schedule that advances communication before copies.
        # Buffer reuse must stop AG2 until scatter0 has consumed buffer0.
        loader.stream.drain()
        loader.scatter_stream.drain()

    monkeypatch.setattr(loader.scatter_stream, "synchronize", finish_copy_stream)
    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
    pcp_load.batch_memcpy_kernel.__getitem__.return_value = scatter
    return loader, sources, targets, gvas, addrs, sizes, read, calls


@pytest.mark.parametrize("family", ["A3", "A5"])
@pytest.mark.parametrize("world,rank", [(2, 0), (8, 7), (16, 15)])
@pytest.mark.parametrize("prefetch", [False, True])
def test_pipeline_byte_exact_with_delayed_streams_and_buffer_reuse(monkeypatch, family, world, rank, prefetch):
    loader, sources, targets, gvas, addrs, sizes, read, calls = pipeline_case(monkeypatch, family, world, rank)
    for step in range(2):
        for source in sources:
            source.add_(23 + step)
        token = object()
        if prefetch:
            loader.prefetch(gvas, sizes, read, token)
        loader.copy(gvas, addrs, sizes, read, token=token)
        for source, target in zip(sources, targets, strict=True):
            assert torch.equal(source, target[16:-16])
            assert torch.all(target[:16] == 0xA5) and torch.all(target[-16:] == 0xA5)
    assert calls.index(("gather", 1)) < calls.index(("scatter", 0)) < calls.index(("gather", 2))
    assert calls.count("metadata_submit") == 6  # Seven windows, three batches per step.
    assert loader.scatter_stream.host_waits == 6 and loader.stream.host_waits == 0
    assert loader.restored_bytes == 2 * int(sizes.sum())


@pytest.mark.parametrize("stage", ["gather", "scatter", "ready_event", "comm_done", "metadata"])
def test_pipeline_partial_enqueue_failure_drains_both_streams(monkeypatch, stage):
    loader, _, _, gvas, addrs, sizes, read, _ = pipeline_case(monkeypatch, "A5", 8, 7)
    if stage == "gather":
        original = pcp_load.dist.all_gather_into_tensor
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected gather")
            return original(*args, **kwargs)

        monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", fail_second)
    elif stage == "scatter":
        monkeypatch.setattr(loader, "_scatter", MagicMock(side_effect=RuntimeError("injected scatter")))
    elif stage == "ready_event":
        monkeypatch.setattr(loader._gather_ready[0], "record", MagicMock(side_effect=RuntimeError("injected event")))
    elif stage == "comm_done":
        monkeypatch.setattr(loader._batch_comm_done, "record", MagicMock(side_effect=RuntimeError("injected event")))
    else:
        original_upload = loader._upload_descriptors

        def fail_upload(*args):
            original_upload(*args)
            raise RuntimeError("injected metadata")

        monkeypatch.setattr(loader, "_upload_descriptors", fail_upload)
    with pytest.raises(RuntimeError, match="injected"):
        loader.copy(gvas, addrs, sizes, read)
    assert loader.stream.completed == len(loader.stream.operations)
    assert loader.scatter_stream.completed == len(loader.scatter_stream.operations)
    assert loader.restored_bytes == 0


def test_device_success_status_is_reused_but_exceptions_force_reset(monkeypatch):
    loader = make_loader(monkeypatch)
    status = SimpleNamespace(value=99, fills=[])

    def fill(value):
        status.value = value
        status.fills.append(value)

    status.fill_ = fill
    status.item = lambda: status.value
    loader._read_status = status
    for _ in range(4):
        loader._check_copy_result(False)
    assert status.fills == [0]
    with pytest.raises(RuntimeError, match="pool read failed"):
        loader._check_copy_result(True)
    loader._check_copy_result(False)
    assert status.fills == [0, 1, 0]

    def fail_reduce(value, **kwargs):
        value.value = 1
        raise RuntimeError("injected collective failure")

    monkeypatch.setattr(pcp_load.dist, "all_reduce", fail_reduce)
    with pytest.raises(RuntimeError, match="injected"):
        loader._check_copy_result(False)
    monkeypatch.setattr(pcp_load.dist, "all_reduce", lambda *a, **kw: None)
    loader._check_copy_result(False)
    assert status.fills == [0, 1, 0, 0]


def test_resident_first_layer_prefetches_next_layer_without_duplicate_task(monkeypatch):
    worker = object.__new__(pool_worker.LayerwiseKVPoolWorker)
    worker.current_layer = worker.next_layer_to_submit = 0
    worker.num_layers = 4
    worker.layer_load_tasks = [[], [object()], [object()], [object()]]
    worker.prefetch_layer_map = {}
    worker.pcp_load = SimpleNamespace(prefetch_enabled=True)
    worker.kv_recv_thread = MagicMock()
    worker._submit_ready_layer_loads()
    worker._submit_ready_layer_loads()
    worker._submit_pcp_h2d_prefetch(1)
    tasks = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
    assert [(x.layer_id, x.pcp_prefetch_only) for x in tasks] == [(1, True)]
    assert worker.next_layer_to_submit == 1
    worker._dflash_draft_start_layer = 2
    worker._submit_pcp_h2d_prefetch(2)
    assert worker.kv_recv_thread.add_request.call_count == 1


def test_failed_prefetch_enqueue_can_retry(monkeypatch):
    worker = object.__new__(pool_worker.LayerwiseKVPoolWorker)
    worker.pcp_load = SimpleNamespace(prefetch_enabled=True)
    worker.num_layers = 2
    worker.layer_load_tasks = [[], [object()]]
    worker.kv_recv_thread = MagicMock()
    worker.kv_recv_thread.add_request.side_effect = RuntimeError("enqueue failed")
    with pytest.raises(RuntimeError, match="enqueue failed"):
        worker._submit_pcp_h2d_prefetch(1)
    worker.kv_recv_thread.add_request.side_effect = None
    worker._submit_pcp_h2d_prefetch(1)
    assert worker._pcp_h2d_prefetch_layer == 1
