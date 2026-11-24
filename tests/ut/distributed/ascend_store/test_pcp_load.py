# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the production restore orchestration with real CPU byte buffers."""

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pcp_load


def make_loader(monkeypatch, rank=0, world_size=2):
    loader = object.__new__(pcp_load.PcpLayerwiseLoad)
    loader.rank = rank
    loader.world_size = world_size
    loader.device = torch.device("cpu")
    loader.local = torch.full((pcp_load.SHARD_CAPACITY,), 0xFA, dtype=torch.uint8)
    loader.gathered = torch.full((pcp_load.SHARD_CAPACITY * world_size,), 0xFB, dtype=torch.uint8)
    loader.control_group = object()
    loader.device_group = object()
    loader.stream = MagicMock()
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


def test_staging_is_aligned_and_communicators_are_separate(monkeypatch):
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda *a, **kw: real_device("cpu"))
    monkeypatch.setattr(torch.npu, "Stream", lambda **kw: MagicMock())
    monkeypatch.setattr(pcp_load, "create_hccl_pg_options", lambda name: SimpleNamespace(hccl_config={}))
    create_group = MagicMock(side_effect=[object(), object()])
    destroy_group = MagicMock()
    monkeypatch.setattr(pcp_load.dist, "new_group", create_group)
    monkeypatch.setattr(pcp_load.dist, "destroy_process_group", destroy_group)
    store = SimpleNamespace(device_id=0, register_buffer=MagicMock(), unregister_buffer=MagicMock())
    group = SimpleNamespace(ranks=[4, 5], world_size=2, rank_in_group=1)
    loader = pcp_load.PcpLayerwiseLoad(store, group)
    assert loader.local.data_ptr() % (2 * 1024 * 1024) == 0
    assert loader.local.numel() == pcp_load.SHARD_CAPACITY
    assert loader.gathered.numel() == pcp_load.SHARD_CAPACITY * group.world_size
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
