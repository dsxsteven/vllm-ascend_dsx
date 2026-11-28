# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the production restore orchestration with real CPU byte buffers."""

import ctypes
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pcp_load, pcp_load_plan


def make_loader(monkeypatch, rank=0, world_size=2):
    loader = object.__new__(pcp_load.PcpLayerwiseLoad)
    loader.rank = rank
    loader.world_size = world_size
    loader.device = torch.device("cpu")
    loader._plan_cache = OrderedDict()
    loader._read_status = None
    loader._status_needs_reset = True
    loader._control_status = torch.empty((), dtype=torch.int32)
    loader.prefetch_enabled = False
    loader.overlap_enabled = False
    loader.batch_windows = 1
    loader._prepared = None
    loader._descriptor_capacity = 0
    loader._host_descriptors = loader._device_descriptors = None
    loader.copy_block_bytes = pcp_load.get_current_hardware_profile().pcp_kv_scatter_block_bytes
    loader.local = torch.full((pcp_load.SHARD_CAPACITY,), 0xFA, dtype=torch.uint8)
    loader.gathered = torch.full((pcp_load.SHARD_CAPACITY * world_size,), 0xFB, dtype=torch.uint8)
    loader.gathered_buffers = [loader.gathered]
    loader.locals = [loader.local]
    loader.control_group = object()
    loader.device_group = object()
    loader.stream = MagicMock()
    loader.scatter_stream = loader.stream
    loader._gather_ready = [MagicMock() for _ in range(pcp_load._GATHER_BUFFERS)]
    loader._scatter_done = [MagicMock() for _ in range(pcp_load._GATHER_BUFFERS)]
    loader._batch_comm_done = MagicMock()
    loader.h2d_bytes = 0
    loader.restored_bytes = 0
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(pcp_load.dist, "all_reduce", lambda *a, **k: None)

    def scatter(src, dst, sizes, **kwargs):
        for source, target, size in zip(src.tolist(), dst.tolist(), sizes.tolist(), strict=True):
            ctypes.memmove(target, source, size)

    kernel = MagicMock()
    kernel.__getitem__.return_value = scatter
    monkeypatch.setattr(pcp_load, "batch_memcpy_kernel", kernel)
    return loader


@pytest.mark.parametrize("world_size,rank", [(2, 0), (2, 1), (8, 3), (16, 0), (16, 15)])
@pytest.mark.parametrize("lengths", [[17, 0, 31], [4097, 73, 16384]])
def test_restore_and_reuse_are_byte_exact(monkeypatch, world_size, rank, lengths):
    loader = make_loader(monkeypatch, rank, world_size)
    sizes = np.asarray(lengths, dtype=np.int64)
    sources = [torch.arange(n, dtype=torch.int64).to(torch.uint8) for n in lengths]
    destinations = [torch.full((n + 32,), 0xA5, dtype=torch.uint8) for n in lengths]
    gvas = np.array([tensor.data_ptr() for tensor in sources], dtype=np.int64)
    addrs = np.array([tensor.data_ptr() + 16 for tensor in destinations], dtype=np.int64)

    def from_pool(srcs, dsts, counts):
        for src, dst, count in zip(srcs, dsts, counts, strict=True):
            ctypes.memmove(int(dst), int(src), int(count))
        return 0

    def all_gather(output, local, *, group):
        assert group is loader.device_group
        expected = torch.cat(sources)
        output.zero_()
        output[: len(expected)].copy_(expected)
        # Peers provide the reference bytes, but this rank must use the bytes
        # the production copy actually fetched, including its padding.
        offset = rank * local.numel()
        output[offset : offset + local.numel()].copy_(local)

    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", all_gather)
    for step in range(2):
        for tensor in sources:
            tensor.add_(11 + step)
        loader.copy(gvas, addrs, sizes, from_pool)
        for source, target in zip(sources, destinations, strict=True):
            assert torch.equal(target[16:-16], source)
            assert torch.all(target[:16] == 0xA5)
            assert torch.all(target[-16:] == 0xA5)
    assert loader.restored_bytes == 2 * sum(lengths)
    assert loader.stream.synchronize.call_count >= 2


@pytest.mark.parametrize("remote_failure", [False, True])
def test_any_pool_failure_prevents_collective_publication(monkeypatch, remote_failure):
    loader = make_loader(monkeypatch)
    gather = MagicMock()
    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
    if remote_failure:
        monkeypatch.setattr(pcp_load.dist, "all_reduce", lambda status, **kw: status.fill_(1))

    def from_pool(*args):
        if remote_failure:
            return 0
        raise RuntimeError("injected backend read failure")

    with pytest.raises(RuntimeError, match="pool read failed"):
        loader.copy(np.array([1]), np.array([2]), np.array([7]), from_pool)
    gather.assert_not_called()
    assert loader.restored_bytes == 0


def test_rank_plan_mismatch_is_rejected(monkeypatch):
    loader = make_loader(monkeypatch)

    def gather_plan(output, value, **kwargs):
        output[:] = [value, "different selection"]

    monkeypatch.setattr(pcp_load.dist, "all_gather_object", gather_plan)
    with pytest.raises(RuntimeError, match="plans differ"):
        loader.validate_plan("logical-key-and-block-digest")


def test_plan_digest_tracks_logical_selection_not_virtual_addresses():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker

    worker = object.__new__(LayerwiseKVPoolWorker)
    worker.pcp_load = MagicMock()
    worker.group_block_len = {0: [16, 8]}
    shared = SimpleNamespace(
        req_ids=["r"], load_keys=["key-b", "key-a"], block_ids_arr=np.array([7, 2]), block_gvas_arr=np.array([100, 200])
    )
    task = SimpleNamespace(shared_block_data=shared, group_id=0, layer_idx_in_group=0)
    worker.layer_load_tasks = [[task], []]

    def digest():
        worker._validate_pcp_load_plan()
        return worker.pcp_load.validate_plan.call_args.args[0]

    first = digest()
    shared.block_gvas_arr = np.array([10000, 20000])
    shared.load_keys.reverse()
    assert digest() == first
    shared.block_ids_arr = np.array([7, 3])
    assert digest() != first
    shared.block_ids_arr = np.array([7, 2])
    worker.group_block_len = {0: [8, 16]}
    assert digest() != first


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("family", ["A3", "A5"])
def test_staging_is_aligned_and_communicators_are_separate(monkeypatch, prefetch, overlap, family):
    profile = get_hardware_profile(getattr(AscendDeviceType, family))
    monkeypatch.setattr(pcp_load, "get_current_hardware_profile", lambda: profile)
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda *a, **kw: real_device("cpu"))
    monkeypatch.setattr(torch.npu, "Stream", lambda **kw: MagicMock())
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(pcp_load, "create_hccl_pg_options", lambda name: SimpleNamespace(hccl_config={}))
    create_group = MagicMock(side_effect=[object(), object()])
    destroy_group = MagicMock()
    monkeypatch.setattr(pcp_load.dist, "new_group", create_group)
    monkeypatch.setattr(pcp_load.dist, "destroy_process_group", destroy_group)
    store = SimpleNamespace(device_id=0, register_buffer=MagicMock(), unregister_buffer=MagicMock())
    group = SimpleNamespace(ranks=[4, 5], world_size=2, rank_in_group=1)
    loader = pcp_load.PcpLayerwiseLoad(store, group, prefetch=prefetch, overlap=overlap)
    assert loader.overlap_enabled is overlap
    assert len(loader.locals) == (3 if prefetch or overlap else 1)
    assert all(local.data_ptr() % (2 * 1024 * 1024) == 0 for local in loader.locals)
    assert loader.local.data_ptr() % (2 * 1024 * 1024) == 0
    assert loader.local.numel() == pcp_load.SHARD_CAPACITY
    assert loader.gathered.numel() == pcp_load.SHARD_CAPACITY * group.world_size
    assert len(loader.gathered_buffers) == (3 if overlap else 1)
    assert len({buffer.data_ptr() for buffer in loader.gathered_buffers}) == len(loader.gathered_buffers)
    assert len(loader._gather_ready) == len(loader._scatter_done) == (3 if overlap else 0)
    assert (loader.scatter_stream is not loader.stream) is overlap
    assert loader.control_group is not loader.device_group
    assert [call.kwargs["backend"] for call in create_group.call_args_list] == ["gloo", "hccl"]
    assert all(call.kwargs["ranks"] == [4, 5] for call in create_group.call_args_list)
    assert create_group.call_args_list[1].kwargs["pg_options"].hccl_config["hccl_buffer_size"] == 64
    loader.close()
    loader.close()
    store.unregister_buffer.assert_called_once_with(*store.register_buffer.call_args.args)
    assert destroy_group.call_count == 2


@pytest.mark.parametrize(
    "override",
    [
        {"backend_name": "mooncake"},
        {"layerwise_offload": False},
        {"use_mla": False},
        {"dcp_size": 2},
        {"pp_size": 2},
        {"use_mamba": True},
        {"use_kvpp": True},
        {"has_shared_kv": True},
    ],
)
def test_unsupported_replication_layout_is_rejected(monkeypatch, override):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker

    attributes = (
        dict(
            pcp_size=16,
            backend_name="memcache",
            layerwise_offload=True,
            use_mla=True,
            dcp_size=1,
            pp_size=1,
            use_mamba=False,
            use_kvpp=False,
            has_shared_kv=False,
        )
        | override
    )

    def fake_base(self, *args):
        self.__dict__.update(attributes)

    monkeypatch.setattr(pool_worker.BaseKVPoolWorker, "__init__", fake_base)
    monkeypatch.setattr(pool_worker.LayerwiseKVPoolWorker, "_init_layerwise_config", lambda self: None)
    monkeypatch.setattr(pool_worker, "get_ascend_config", lambda: SimpleNamespace(kv_pool_pcp_h2d_allgather=True))
    config = SimpleNamespace(additional_config={"kv_pool_pcp_h2d_allgather": True})
    with pytest.raises(ValueError, match="replicated MLA KV"):
        pool_worker.LayerwiseKVPoolWorker(config, True)


@pytest.mark.parametrize(
    "flag,pcp_size,enabled", [(None, 16, False), (False, 16, False), (True, 1, False), (True, 16, True)]
)
def test_flag_is_opt_in_and_pcp_one_uses_direct_copy(monkeypatch, flag, pcp_size, enabled):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker

    def fake_base(self, *args):
        self.__dict__.update(
            pcp_size=pcp_size,
            backend_name="memcache",
            layerwise_offload=True,
            use_mla=True,
            dcp_size=1,
            pp_size=1,
            use_mamba=False,
            use_kvpp=False,
            has_shared_kv=False,
        )

    getter = MagicMock(return_value=SimpleNamespace(kv_pool_pcp_h2d_allgather=flag))
    monkeypatch.setattr(pool_worker.BaseKVPoolWorker, "__init__", fake_base)
    monkeypatch.setattr(pool_worker.LayerwiseKVPoolWorker, "_init_layerwise_config", lambda self: None)
    monkeypatch.setattr(pool_worker, "get_ascend_config", getter)
    config = SimpleNamespace(additional_config={} if flag is None else {"kv_pool_pcp_h2d_allgather": flag})
    worker = pool_worker.LayerwiseKVPoolWorker(config, True)
    assert worker.pcp_load_enabled is enabled
    assert worker.pcp_load is None
    if flag is None:
        getter.assert_not_called()


@pytest.mark.parametrize("sharded", [False, True])
def test_pcp_collectives_are_not_prefetched_across_model_layers(monkeypatch, sharded):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker

    worker = object.__new__(pool_worker.LayerwiseKVPoolWorker)
    worker.pcp_load = object() if sharded else None
    worker.current_layer = worker.next_layer_to_submit = 2
    worker.num_layers = 8
    worker.num_prefetch_layers = 3
    worker.layer_load_tasks = [[object()] for _ in range(8)]
    worker.prefetch_layer_map = {5: (2,)}
    worker.kv_recv_thread = MagicMock()
    gate = object()
    monkeypatch.setattr(pool_worker, "get_attention_compute_start_gate", lambda: gate)
    worker._submit_ready_layer_loads()
    tasks = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
    assert [task.layer_id for task in tasks] == ([2] if sharded else [2, 3, 4, 5])
    assert tasks[0].attention_start_gate is None
    if sharded:
        assert worker.next_layer_to_submit == 3
    else:
        assert all(task.attention_start_gate is gate for task in tasks[1:])
        assert tasks[-1].wait_for_save_layers == (2,)


@pytest.mark.parametrize("sharded", [False, True])
def test_pcp_demand_load_fences_model_stream_before_queueing(monkeypatch, sharded):
    import threading

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker

    worker = object.__new__(pool_worker.LayerwiseKVPoolWorker)
    worker.pcp_load = object() if sharded else None
    worker.current_layer = worker.next_layer_to_submit = 0
    worker.num_layers = 1
    worker.use_kvpp = False
    worker.layer_load_tasks = [[object()]]
    worker.prefetch_layer_map = {}
    worker.layer_load_finished_events = [threading.Event()]
    worker.layer_load_finished_events[0].set()
    worker.kv_recv_thread = MagicMock()
    events = []
    boundary = MagicMock()
    boundary.record.side_effect = lambda: events.append("record")
    boundary.synchronize.side_effect = lambda: events.append("synchronize")
    monkeypatch.setattr(torch.npu, "Event", lambda: boundary)
    worker._submit_ready_layer_loads = lambda: events.append("enqueue")
    worker.wait_for_layer_load()
    assert events == (["record", "synchronize", "enqueue"] if sharded else ["enqueue"])
    assert not worker.layer_load_finished_events[0].is_set()


def test_transfer_plan_cache_snapshots_lengths_and_is_bounded(monkeypatch):
    loader = make_loader(monkeypatch)
    sizes = np.array([17, 0, 31], dtype=np.int64)
    original = loader._get_transfer_plan(sizes)
    assert loader._get_transfer_plan(sizes.copy()) is original
    sizes[0] += 1
    changed = loader._get_transfer_plan(sizes)
    assert changed is not original
    assert original[0][1] == 48 and changed[0][1] == 49
    loader._get_transfer_plan(np.array([17, 0, 31], dtype=np.int32))
    assert len(loader._plan_cache) == 2
    assert loader._get_transfer_plan(np.array([17, 0, 31], dtype=np.int64)) is not original


@pytest.mark.parametrize("device_status", [False, True])
def test_device_status_failure_is_not_stale_and_uses_private_group(monkeypatch, device_status):
    loader = make_loader(monkeypatch)
    loader._read_status = torch.empty((), dtype=torch.int32) if device_status else None
    peer_failed = False

    def reduce(status, *, op, group):
        assert group is (loader.device_group if device_status else loader.control_group)
        if peer_failed:
            status.fill_(1)

    monkeypatch.setattr(pcp_load.dist, "all_reduce", reduce)
    loader._check_copy_result(False)
    with pytest.raises(RuntimeError, match="pool read failed"):
        loader._check_copy_result(True)
    loader._check_copy_result(False)
    peer_failed = True
    with pytest.raises(RuntimeError, match="pool read failed"):
        loader._check_copy_result(False)
    peer_failed = False
    loader._check_copy_result(False)


@pytest.mark.parametrize("world_size,rank", [(2, 0), (2, 1), (8, 7), (16, 0), (16, 15)])
@pytest.mark.parametrize("prefetch", [False, True])
def test_batched_restore_preserves_async_buffer_lifetimes(monkeypatch, world_size, rank, prefetch):
    # Small shard capacity exercises seven windows and three batches cheaply.
    monkeypatch.setattr(pcp_load, "SHARD_CAPACITY", 512)
    monkeypatch.setattr(
        pcp_load,
        "shard_window",
        lambda total, world, local: pcp_load_plan.shard_window(total, world, local, capacity=512),
    )
    loader = make_loader(monkeypatch, rank, world_size)
    loader.prefetch_enabled = True
    loader.batch_windows = 3
    loader.locals = [torch.full((512,), 0xFA, dtype=torch.uint8) for _ in range(3)]
    loader.local = loader.locals[0]
    lengths = [101, 0, world_size * 512 * 6 - 101, 7]
    sizes = np.asarray(lengths, dtype=np.int64)
    sources = [torch.arange(n, dtype=torch.int64).to(torch.uint8) for n in lengths]
    targets = [torch.full((n + 32,), 0xA5, dtype=torch.uint8) for n in lengths]
    gvas = np.array([x.data_ptr() for x in sources], dtype=np.int64)
    addrs = np.array([x.data_ptr() + 16 for x in targets], dtype=np.int64)
    plan = loader._get_transfer_plan(sizes)
    assert len(plan) == 7
    calls = []
    queue = []
    window_index = 0

    def read(srcs, dsts, counts):
        calls.append("read")
        assert not queue, "staging must not be overwritten while an earlier batch uses it"
        for src, dst, count in zip(srcs, dsts, counts, strict=True):
            ctypes.memmove(int(dst), int(src), int(count))
        return 0

    def gather(output, local, *, group):
        nonlocal window_index
        assert group is loader.device_group
        window = plan[window_index % len(plan)]
        window_index += 1
        calls.append("gather")

        def execute():
            start, end, shard_bytes, valid, _, _ = window
            reference = torch.cat(sources)[start:end]
            offset = rank * shard_bytes
            assert torch.equal(local[:valid], reference[offset : offset + valid])
            assert torch.all(local[valid:] == 0)
            output.zero_()
            output[: len(reference)].copy_(reference)
            output[offset : offset + shard_bytes].copy_(local)

        queue.append(execute)

    def scatter(src, dst, counts, **kw):
        # Retain views, rather than snapshotting: early descriptor reuse would
        # redirect these delayed writes and corrupt the guard/reference checks.
        def execute():
            for source, target, count in zip(src.tolist(), dst.tolist(), counts.tolist(), strict=True):
                ctypes.memmove(target, source, count)

        queue.append(execute)

    def synchronize():
        calls.append("sync")
        for operation in queue:
            operation()
        queue.clear()

    loader.stream.synchronize.side_effect = synchronize
    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(pcp_load.dist, "all_reduce", lambda *a, **kw: calls.append("status"))
    pcp_load.batch_memcpy_kernel.__getitem__.return_value = scatter
    descriptor_ptr = None
    for step in range(2):
        token = object()
        for source in sources:
            source.add_(19 + step)
        before = len(calls)
        if prefetch:
            loader.prefetch(gvas, sizes, read, token)
            assert calls[before:] == ["read"] * 3
        loader.copy(gvas, addrs, sizes, read, token=token)
        assert calls[before:].count("read") == 7
        assert calls[before:].count("gather") == 7
        assert calls[before:].count("status") == 3
        assert calls[before:].count("sync") == 3
        assert not queue
        for source, target in zip(sources, targets, strict=True):
            assert torch.equal(source, target[16:-16])
            assert torch.all(target[:16] == 0xA5) and torch.all(target[-16:] == 0xA5)
        current = loader._device_descriptors.data_ptr()
        assert descriptor_ptr is None or current == descriptor_ptr
        descriptor_ptr = current
    assert loader.restored_bytes == 2 * sum(lengths)


@pytest.mark.parametrize("remote_failure", [False, True])
def test_prefetch_failure_is_deferred_until_demand_and_recovers(monkeypatch, remote_failure):
    loader = make_loader(monkeypatch)
    loader.prefetch_enabled = True
    token = object()
    gvas, sizes = np.array([1]), np.array([7])
    gather, status = MagicMock(), MagicMock()
    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(pcp_load.dist, "all_reduce", status)
    read = MagicMock(side_effect=None if remote_failure else RuntimeError("local prefetch failure"), return_value=0)
    loader.prefetch(gvas, sizes, read, token)
    gather.assert_not_called()
    status.assert_not_called()
    if remote_failure:
        status.side_effect = lambda value, **kw: value.fill_(1)
    with pytest.raises(RuntimeError, match="pool read failed"):
        loader.copy(gvas, np.array([2]), sizes, read, token=token)
    assert read.call_count == 1
    gather.assert_not_called()
    assert loader._prepared is None
    assert loader.restored_bytes == 0
    status.side_effect = None
    loader._check_copy_result(False)


@pytest.mark.parametrize("mismatch", ["token", "gva", "size"])
def test_prefetch_never_reuses_stale_pool_content(monkeypatch, mismatch):
    loader = make_loader(monkeypatch)
    loader.prefetch_enabled = True
    token = object()
    gvas, sizes = np.array([1]), np.array([7])
    read = MagicMock(return_value=0)
    loader.prefetch(gvas, sizes, read, token)
    if mismatch == "token":
        token = object()
    elif mismatch == "gva":
        gvas[0] += 100
    else:
        sizes[0] += 1
    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", MagicMock())
    monkeypatch.setattr(loader, "_scatter", MagicMock())
    loader.copy(gvas, np.array([2]), sizes, read, token=token)
    assert read.call_count == 2
    assert loader._prepared is None


def test_step_validation_discards_unused_prefetch(monkeypatch):
    loader = make_loader(monkeypatch)
    loader.prefetch_enabled = True
    loader.prefetch(np.array([1]), np.array([7]), lambda *a: 0, object())
    monkeypatch.setattr(
        pcp_load.dist, "all_gather_object", lambda output, value, **kw: output.__setitem__(slice(None), [value] * 2)
    )
    loader.validate_plan("next-step")
    assert loader._prepared is None
    loader.stream.synchronize.assert_called_once_with()


def test_lookahead_is_local_only_and_does_not_advance_demand_cursor(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker

    worker = object.__new__(pool_worker.LayerwiseKVPoolWorker)
    worker.pcp_load = SimpleNamespace(prefetch_enabled=True)
    worker.current_layer = worker.next_layer_to_submit = 2
    worker.num_layers = 5
    worker.layer_load_tasks = [[object()] for _ in range(5)]
    worker.prefetch_layer_map = {3: (1,)}
    worker.kv_recv_thread = MagicMock()
    worker._submit_ready_layer_loads()
    worker._submit_ready_layer_loads()
    tasks = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
    assert [(x.layer_id, x.pcp_prefetch_only) for x in tasks] == [(2, False), (3, True)]
    assert tasks[1].wait_for_save_layers == ()
    assert tasks[1].attention_start_gate is None
    assert worker.next_layer_to_submit == 3
    worker.current_layer = 3
    worker._submit_ready_layer_loads()
    demand = worker.kv_recv_thread.add_request.call_args_list[2].args[0]
    assert demand.layer_id == 3 and not demand.pcp_prefetch_only
    assert demand.wait_for_save_layers == (1,)
    assert demand.transfer_tasks is tasks[1].transfer_tasks


@pytest.mark.parametrize("failing_window", [0, 1, 2])
def test_one_failed_read_rejects_the_entire_unpublished_batch(monkeypatch, failing_window):
    monkeypatch.setattr(pcp_load, "SHARD_CAPACITY", 512)
    monkeypatch.setattr(
        pcp_load,
        "shard_window",
        lambda total, world, rank: pcp_load_plan.shard_window(total, world, rank, capacity=512),
    )
    loader = make_loader(monkeypatch)
    loader.batch_windows = 3
    loader.locals = [torch.empty(512, dtype=torch.uint8) for _ in range(3)]
    reads = 0

    def read(*args):
        nonlocal reads
        result = -1 if reads == failing_window else 0
        reads += 1
        return result

    gather = MagicMock()
    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
    with pytest.raises(RuntimeError, match="pool read failed"):
        loader.copy(np.array([1]), np.array([2]), np.array([3072]), read)
    assert reads == 3
    gather.assert_not_called()
    assert loader.restored_bytes == 0
    loader.stream.synchronize.assert_called_once_with()


@pytest.mark.parametrize("field", [1, 2, 3, 4, 5])
def test_rank_pipeline_configuration_mismatch_is_rejected(monkeypatch, field):
    loader = make_loader(monkeypatch)

    def gather(output, identity, **kwargs):
        different = list(identity)
        different[field] = not identity[field] if isinstance(identity[field], bool) else identity[field] + 1
        output[:] = [identity, tuple(different)]

    monkeypatch.setattr(pcp_load.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="plans differ"):
        loader.validate_plan("identical-logical-selection")
