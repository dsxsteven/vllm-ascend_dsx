# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU byte buffers exercise production single-gather orchestration."""

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pcp_load


def cpu_scatter(pointers, gathered, result, payload, world, trailer, block):
    flags = gathered[torch.arange(world) * (payload + trailer) + payload]
    result[0] = int(bool(flags.any()))
    if not result[0]:
        for src, dst, length in zip(*pointers.tolist(), strict=True):
            ctypes.memmove(dst, src, length)


def make_loader(monkeypatch, rank=0, world_size=2, family="A3", prefetch=False, overlap=False):
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda *args, **kw: real_device("cpu"))
    monkeypatch.setattr(torch.npu, "Stream", lambda **kw: MagicMock())
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(
        pcp_load, "get_current_hardware_profile", lambda: get_hardware_profile(getattr(AscendDeviceType, family))
    )
    monkeypatch.setattr(pcp_load, "create_hccl_pg_options", lambda _: SimpleNamespace(hccl_config={}))
    monkeypatch.setattr(pcp_load.dist, "new_group", MagicMock(side_effect=[object(), object()]))
    monkeypatch.setattr(pcp_load.dist, "destroy_process_group", MagicMock())
    monkeypatch.setattr(
        pcp_load.dist, "all_gather_object", lambda out, value, **kw: out.__setitem__(slice(None), [value] * world_size)
    )
    monkeypatch.setattr(pcp_load.dist, "all_reduce", MagicMock(side_effect=AssertionError("No standalone AllReduce")))
    monkeypatch.setattr(pcp_load, "checked_pcp_scatter", cpu_scatter)
    store = SimpleNamespace(device_id=0, register_buffer=MagicMock(), unregister_buffer=MagicMock())
    group = SimpleNamespace(ranks=list(range(world_size)), world_size=world_size, rank_in_group=rank)
    return pcp_load.PcpLayerwiseLoad(store, group, prefetch=prefetch, overlap=overlap)


def read_bytes(srcs, dsts, sizes):
    for src, dst, size in zip(srcs, dsts, sizes, strict=True):
        ctypes.memmove(int(dst), int(src), int(size))
    return 0


def byte_case(loader, lengths):
    sizes = np.array(lengths, dtype=np.int64)
    sources = [(torch.arange(n, dtype=torch.int64) * 37 + i * 11).to(torch.uint8) for i, n in enumerate(lengths)]
    targets = [torch.full((n + 32,), 0xA5, dtype=torch.uint8) for n in lengths]
    gvas = np.array([t.data_ptr() for t in sources], dtype=np.int64)
    addrs = np.array([t.data_ptr() + 16 for t in targets], dtype=np.int64)
    return sources, targets, gvas, addrs, sizes


def install_gather(monkeypatch, loader, sources, failed_rank=None):
    calls = []

    def gather(output, local, *, group):
        assert group is loader.device_group
        payload = local.numel() - pcp_load._STATUS_BYTES
        assert output.numel() == local.numel() * loader.world_size
        expected = torch.cat(sources)
        rows = output.view(loader.world_size, local.numel())
        rows.zero_()
        for rank, row in enumerate(rows):
            part = expected[rank * payload : (rank + 1) * payload]
            row[: part.numel()].copy_(part)
        rows[loader.rank].copy_(local)
        if failed_rank is not None:
            rows[failed_rank, payload] = 1
        calls.append(payload)

    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
    return calls


@pytest.mark.parametrize("family", ["A3", "A5"])
@pytest.mark.parametrize("world,rank", [(2, 0), (3, 2), (8, 7), (16, 0), (16, 15)])
@pytest.mark.parametrize("lengths", [[1], [17, 0, 31], [4097, 73, 16384], [65537, 512, 131072]])
@pytest.mark.parametrize("prefetch", [False, True])
def test_single_gather_restore_and_reuse(monkeypatch, family, world, rank, lengths, prefetch):
    loader = make_loader(monkeypatch, rank, world, family, prefetch, True)
    sources, targets, gvas, addrs, sizes = byte_case(loader, lengths)
    calls = install_gather(monkeypatch, loader, sources)
    for step in range(2):
        for source in sources:
            source.add_(11 + step)
        token = object()
        if prefetch:
            loader.prefetch(gvas, sizes, read_bytes, token)
        loader.copy(gvas, addrs, sizes, read_bytes, token=token)
        for source, target in zip(sources, targets, strict=True):
            assert torch.equal(source, target[16:-16])
            assert torch.all(target[:16] == 0xA5) and torch.all(target[-16:] == 0xA5)
    assert len(calls) == 2
    assert loader.restored_bytes == 2 * sum(lengths)
    assert loader.local.data_ptr() % (2 * 1024 * 1024) == 0
    assert loader.store.register_buffer.call_count == 1
    pcp_load.dist.all_reduce.assert_not_called()
    loader.close()
    loader.close()
    assert loader.store.unregister_buffer.call_count == 1


@pytest.mark.parametrize("family", ["A3", "A5"])
@pytest.mark.parametrize("failure", ["return", "exception", "peer", "prefetch_error"])
def test_failure_participates_but_never_publishes_and_recovers(monkeypatch, family, failure):
    loader = make_loader(monkeypatch, family=family, prefetch=True, overlap=True)
    sources, targets, gvas, addrs, sizes = byte_case(loader, [101, 0, 2049])
    calls = install_gather(monkeypatch, loader, sources, 1 if failure == "peer" else None)

    def read(*args):
        read_bytes(*args)
        if failure == "exception":
            raise RuntimeError("injected backend failure")
        return -1 if failure == "return" else 0

    token = object()
    if failure == "prefetch_error":
        loader.defer_prefetch_error(token, ValueError("injected metadata failure"))
    else:
        loader.prefetch(gvas, sizes, read, token)
    with pytest.raises(RuntimeError, match="pool read failed"):
        loader.copy(gvas, addrs, sizes, read, token=token)
    assert len(calls) == 1
    assert all(torch.all(target == 0xA5) for target in targets)
    assert loader.restored_bytes == 0
    install_gather(monkeypatch, loader, sources)
    loader.copy(gvas, addrs, sizes, read_bytes)
    for source, target in zip(sources, targets, strict=True):
        assert torch.equal(source, target[16:-16])


def test_empty_restore_does_not_allocate_or_communicate(monkeypatch):
    loader = make_loader(monkeypatch)
    gather = MagicMock()
    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
    empty = np.array([], dtype=np.int64)
    loader.copy(empty, empty, empty, read_bytes)
    gather.assert_not_called()
    loader.store.register_buffer.assert_not_called()


@pytest.mark.parametrize("family", ["A3", "A5"])
def test_payload_above_old_32_mib_limit_is_one_gather(monkeypatch, family):
    loader = make_loader(monkeypatch, family=family, prefetch=True, overlap=True)
    length = 70 * 1024 * 1024 + 17
    source = torch.arange(256, dtype=torch.int64).to(torch.uint8).repeat((length + 255) // 256)[:length]
    target = torch.full((length + 32,), 0xA5, dtype=torch.uint8)
    sizes = np.array([length], dtype=np.int64)
    gvas, addrs = np.array([source.data_ptr()]), np.array([target.data_ptr() + 16])
    calls = install_gather(monkeypatch, loader, [source])
    token = object()
    loader.prefetch(gvas, sizes, read_bytes, token)
    loader.copy(gvas, addrs, sizes, read_bytes, token=token)
    assert len(calls) == 1 and calls[0] > 32 * 1024 * 1024
    assert torch.equal(target[16:-16], source)
    assert torch.all(target[:16] == 0xA5) and torch.all(target[-16:] == 0xA5)
    pcp_load.dist.all_reduce.assert_not_called()


def test_dynamic_growth_and_smaller_reuse(monkeypatch):
    loader = make_loader(monkeypatch)
    for length in [4097, 5 * 1024 * 1024 + 1, 17]:
        sources, targets, gvas, addrs, sizes = byte_case(loader, [length])
        calls = install_gather(monkeypatch, loader, sources)
        loader.copy(gvas, addrs, sizes, read_bytes)
        assert len(calls) == 1
        assert torch.equal(targets[0][16:-16], sources[0])
    assert loader.store.register_buffer.call_count == 2
    assert loader.store.unregister_buffer.call_count == 1
    assert loader._capacity == 4 * 1024 * 1024


def test_failed_growth_keeps_old_registered_storage(monkeypatch):
    loader = make_loader(monkeypatch)
    loader._ensure_capacity(1024)
    old_pointer = loader.local.data_ptr()
    monkeypatch.setattr(
        pcp_load.dist,
        "all_gather_object",
        lambda out, value, **kw: out.__setitem__(slice(None), [value, (value[0], False)]),
    )
    with pytest.raises(RuntimeError, match="allocation/registration"):
        loader._ensure_capacity(3 * 1024 * 1024)
    assert loader.local.data_ptr() == old_pointer
    assert loader.store.unregister_buffer.call_args.args[0] != [old_pointer]


@pytest.mark.parametrize("mismatch", ["token", "size", "gva"])
def test_stale_prefetch_is_not_reused(monkeypatch, mismatch):
    loader = make_loader(monkeypatch, prefetch=True)
    sources, targets, gvas, addrs, sizes = byte_case(loader, [4097])
    install_gather(monkeypatch, loader, sources)
    token = object()
    loader.prefetch(gvas, sizes, read_bytes, token)
    sources[0].add_(29)
    if mismatch == "token":
        token = object()
    elif mismatch == "size":
        sizes = sizes - 1
    else:
        sources[0] = sources[0].clone()
        gvas = np.array([sources[0].data_ptr()])
    loader.copy(gvas, addrs, sizes, read_bytes, token=token)
    assert torch.equal(targets[0][16 : 16 + sizes[0]], sources[0][: sizes[0]])


def test_rank_plan_mismatch_is_rejected(monkeypatch):
    loader = make_loader(monkeypatch)
    monkeypatch.setattr(
        pcp_load.dist, "all_gather_object", lambda out, value, **kw: out.__setitem__(slice(None), [value, "different"])
    )
    with pytest.raises(RuntimeError, match="plans differ"):
        loader.validate_plan("digest")


def test_cached_plan_reuses_ranges_but_tracks_mutated_sizes(monkeypatch):
    loader = make_loader(monkeypatch)
    sizes = np.array([513, 1025], dtype=np.int64)
    first = loader._get_transfer_plan(sizes)
    assert loader._get_transfer_plan(sizes.copy()) is first
    sizes[1] += 512
    second = loader._get_transfer_plan(sizes)
    assert second is not first and second[0][1] == int(sizes.sum())
    # The cache contains relative ranges only; production copy attaches the
    # current destination addresses, including after an allocator reuses them.
    for lengths in ([513, 1537], [513, 1537]):
        sources, targets, gvas, addrs, current_sizes = byte_case(loader, lengths)
        install_gather(monkeypatch, loader, sources)
        loader.copy(gvas, addrs, current_sizes, read_bytes)
        assert all(torch.equal(source, target[16:-16]) for source, target in zip(sources, targets, strict=True))
    loader._get_transfer_plan(np.array([7], dtype=np.int64))
    assert len(loader._plan_cache) == 2


@pytest.mark.parametrize("field", [1, 2, 3, 4])
def test_wire_capacity_and_mode_mismatch_rejected_before_data_collective(monkeypatch, field):
    loader = make_loader(monkeypatch)

    def gather(out, identity, **kwargs):
        peer = list(identity)
        peer[field] = int(peer[field]) + 1
        out[:] = [identity, tuple(peer)]

    monkeypatch.setattr(pcp_load.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="plans differ"):
        loader.validate_plan("same-logical-keys")


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
