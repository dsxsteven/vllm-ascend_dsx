# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU scheduling/byte tests for bounded PCP collective overlap."""

import ctypes
import queue
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.ut.distributed.ascend_store.test_pcp_load import make_loader
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pcp_load, pool_worker


def make_worker(monkeypatch, layer=2):
    worker = object.__new__(pool_worker.LayerwiseKVPoolWorker)
    worker.current_layer = layer
    worker.next_layer_to_submit = layer + 1
    worker.num_layers = 6
    worker.layer_load_tasks = [[object()] for _ in range(worker.num_layers)]
    worker.layer_load_finished_events = [threading.Event() for _ in range(worker.num_layers)]
    worker.prefetch_layer_map = {3: (0,), 4: (1,), 5: (2,)}
    worker.pcp_load = SimpleNamespace(overlap_enabled=True, prefetch_enabled=True, device="cpu")
    worker.kv_recv_thread = MagicMock()
    worker._pcp_overlap_layer = None
    worker._layer_load_aborted = threading.Event()
    worker._dflash_draft_start_layer = None
    monkeypatch.setattr(torch.npu, "synchronize", MagicMock())
    # The shared CPU fixture mocks the worker's gate helpers. These tests need
    # the real handoff between the worker callback and attention context.
    monkeypatch.setattr(
        pool_worker, "reset_attention_compute_start_gate", attention_fence.reset_attention_compute_start_gate
    )
    monkeypatch.setattr(
        pool_worker, "get_attention_compute_start_gate", attention_fence.get_attention_compute_start_gate
    )
    return worker


def bind_window(worker, layer):
    gate = attention_fence.reset_attention_compute_start_gate()
    gate.on_overlap_start = lambda: worker._start_pcp_overlap_window(layer)
    gate.on_overlap_finish = worker._finish_pcp_overlap_window
    return gate


@pytest.mark.parametrize("pcp_size", [1, 8])
@pytest.mark.parametrize("eager", [False, True])
def test_collective_overlap_requires_eager_execution(monkeypatch, pcp_size, eager):
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

    monkeypatch.setattr(pool_worker.BaseKVPoolWorker, "__init__", fake_base)
    monkeypatch.setattr(pool_worker.LayerwiseKVPoolWorker, "_init_layerwise_config", lambda self: None)
    monkeypatch.setattr(
        pool_worker,
        "get_ascend_config",
        lambda: SimpleNamespace(kv_pool_pcp_h2d_allgather=True, kv_pool_pcp_restore_overlap=True),
    )
    config = SimpleNamespace(
        additional_config={"kv_pool_pcp_h2d_allgather": True}, model_config=SimpleNamespace(enforce_eager=eager)
    )
    if pcp_size > 1 and not eager:
        with pytest.raises(ValueError, match="enforce_eager=True"):
            pool_worker.LayerwiseKVPoolWorker(config, True)
    else:
        worker = pool_worker.LayerwiseKVPoolWorker(config, True)
        assert worker.pcp_load_enabled is (pcp_size > 1)


@pytest.mark.parametrize("consumer_only", [False, True])
def test_normal_consumer_uses_overlap_completion_without_duplicate_restore(monkeypatch, consumer_only):
    worker = make_worker(monkeypatch)
    worker.use_kvpp = False
    worker.kv_role = "kv_consumer" if consumer_only else "kv_both"
    worker.external_slot_release_waiter = None
    worker.layer_load_finished_events[2].set()
    boundary = MagicMock()
    monkeypatch.setattr(torch.npu, "Event", boundary)
    worker.wait_for_layer_load()
    assert worker.current_layer == (3 if consumer_only else 2)
    with attention_fence.attention_transfer_window(allow_pcp_overlap=True):
        worker.layer_load_finished_events[3].set()
    assert worker.next_layer_to_submit == 4
    # The callback must capture L even when a consumer advances its cursor
    # before attention. L+1 then consumes exactly the same receiver event.
    worker.current_layer = 3
    worker.wait_for_layer_load()
    assert not worker.layer_load_finished_events[3].is_set()
    tasks = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
    assert [(task.layer_id, task.pcp_prefetch_only) for task in tasks] == [(3, False), (4, True)]
    # Gate recording may create an event; demand fencing must not wait again.
    boundary.return_value.synchronize.assert_not_called()


def test_restore_only_opens_in_explicit_bounded_window(monkeypatch):
    worker = make_worker(monkeypatch)
    gate = bind_window(worker, 2)
    gate.record()
    with attention_fence.attention_transfer_window():
        pass
    worker.kv_recv_thread.add_request.assert_not_called()
    with attention_fence.attention_transfer_window(allow_pcp_overlap=True):
        tasks = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
        assert [(t.layer_id, t.pcp_prefetch_only) for t in tasks] == [(3, False), (4, True)]
        assert tasks[0].wait_for_save_layers == (0,)
        assert tasks[0].attention_start_gate is None
        assert worker.next_layer_to_submit == 4
        worker.layer_load_finished_events[3].set()
    assert worker._pcp_overlap_layer is None
    assert worker.layer_load_finished_events[3].is_set()
    torch.npu.synchronize.assert_called_once_with("cpu")
    # Opening another compute window in the same layer must not duplicate KV.
    with attention_fence.attention_transfer_window(allow_pcp_overlap=True):
        pass
    assert worker.kv_recv_thread.add_request.call_count == 2


@pytest.mark.parametrize("case", ["current_slot", "future_slot", "draft", "empty", "tail", "submitted"])
def test_ineligible_lookahead_remains_demand_loaded(monkeypatch, case):
    worker = make_worker(monkeypatch)
    if case == "current_slot":
        worker.prefetch_layer_map[3] = (2,)
    elif case == "future_slot":
        worker.prefetch_layer_map[3] = (4,)
    elif case == "draft":
        worker._dflash_draft_start_layer = 3
    elif case == "empty":
        worker.layer_load_tasks[3] = []
    elif case == "tail":
        worker.num_layers = 3
    else:
        worker.next_layer_to_submit = 4
    before = worker.next_layer_to_submit
    bind_window(worker, 2)
    with attention_fence.attention_transfer_window(allow_pcp_overlap=True):
        pass
    worker.kv_recv_thread.add_request.assert_not_called()
    torch.npu.synchronize.assert_not_called()
    assert worker.next_layer_to_submit == before


@pytest.mark.parametrize("failure", ["attention", "receiver", "enqueue_lookahead"])
def test_exception_cannot_leave_collectives_outside_window(monkeypatch, failure):
    worker = make_worker(monkeypatch)
    bind_window(worker, 2)
    if failure == "enqueue_lookahead":

        def fail_lookahead(layer):
            worker.layer_load_finished_events[3].set()
            raise RuntimeError("enqueue_lookahead")

        monkeypatch.setattr(worker, "_submit_pcp_h2d_prefetch", fail_lookahead)
    with (
        pytest.raises(RuntimeError, match=failure),
        attention_fence.attention_transfer_window(allow_pcp_overlap=True),
    ):
        worker.layer_load_finished_events[3].set()
        if failure == "receiver":
            worker.kv_recv_thread.raise_if_failed.side_effect = RuntimeError("receiver")
        else:
            raise RuntimeError("attention")
    assert worker._pcp_overlap_layer is None
    assert worker.layer_load_finished_events[3].is_set()
    if failure == "receiver":
        assert worker._layer_load_aborted.is_set()


def test_window_cancel_and_legacy_callbacks_remain_independent(monkeypatch):
    worker = make_worker(monkeypatch)
    gate = bind_window(worker, 2)
    legacy = []
    gate.on_start = lambda: legacy.append("start")
    gate.on_finish = lambda: legacy.append("finish")
    with attention_fence.attention_transfer_window(allow_pcp_overlap=True):
        worker.layer_load_finished_events[3].set()
    assert legacy == ["start", "finish"]
    gate = bind_window(worker, 3)
    gate.cancel()
    with attention_fence.attention_transfer_window(allow_pcp_overlap=True):
        pass
    assert worker.kv_recv_thread.add_request.call_count == 2


def test_mixed_overlap_flags_are_rejected_before_step(monkeypatch):
    loader = make_loader(monkeypatch)
    loader.overlap_enabled = True

    def gather(output, identity, **kwargs):
        output[:] = [identity, (*identity[:-1], False)]

    monkeypatch.setattr(pcp_load.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="plans differ"):
        loader.validate_plan("same blocks")


@pytest.mark.parametrize("family", ["A3", "A5"])
@pytest.mark.parametrize("world_size", [2, 8, 16])
@pytest.mark.parametrize("include_indexer", [False, True])
def test_byte_exact_restore_executes_while_compute_is_inflight(monkeypatch, family, world_size, include_indexer):
    loader = make_loader(monkeypatch, world_size=world_size, family=family, prefetch=True, overlap=True)
    source = torch.arange(world_size * 1024 + 13, dtype=torch.int64).to(torch.uint8)
    target = torch.full((source.numel() + 32,), 0xA5, dtype=torch.uint8)
    compute_submitted = threading.Event()
    restore_finished = threading.Event()
    errors = []
    calls = []

    def gather(output, local, *, group):
        assert compute_submitted.wait(timeout=5), "model compute was blocked before restore"
        assert group is loader.device_group
        calls.append("allgather_during_compute")
        payload = local.numel() - pcp_load._STATUS_BYTES
        rows = output.view(world_size, local.numel())
        rows.zero_()
        for rank, row in enumerate(rows):
            part = source[rank * payload : (rank + 1) * payload]
            row[: part.numel()].copy_(part)
        rows[loader.rank].copy_(local)

    def read(srcs, dsts, counts):
        for src, dst, count in zip(srcs, dsts, counts, strict=True):
            ctypes.memmove(int(dst), int(src), int(count))
        return 0

    monkeypatch.setattr(pcp_load.dist, "all_gather_into_tensor", gather)
    worker = make_worker(monkeypatch)
    worker.pcp_load = loader
    requests = queue.Queue()
    worker.kv_recv_thread.add_request.side_effect = requests.put

    def receive():
        try:
            task = requests.get(timeout=5)
            assert task.layer_id == 3 and not task.pcp_prefetch_only
            loader.copy(
                np.array([source.data_ptr()]),
                np.array([target.data_ptr() + 16]),
                np.array([source.numel()]),
                read,
            )
            calls.append("scatter_complete")
        except BaseException as error:
            errors.append(error)
        finally:
            worker.layer_load_finished_events[3].set()
            restore_finished.set()

    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    bind_window(worker, 2)
    scope = attention_fence.indexer_attention_transfer_window() if include_indexer else nullcontext()
    with scope:
        if include_indexer:
            attention_fence.start_local_indexer_transfer_overlap()
            calls.append("submit_indexer")
            compute_submitted.set()
        with attention_fence.attention_transfer_window(allow_pcp_overlap=True):
            # Simulate asynchronous model kernels; no KV join between LI/SFA.
            assert worker._pcp_overlap_layer == 3
            calls.append("submit_attention")
            compute_submitted.set()
    calls.append("next_model_collective")
    thread.join(timeout=5)
    assert not thread.is_alive() and not errors, errors
    assert restore_finished.is_set()
    assert calls[0] == ("submit_indexer" if include_indexer else "submit_attention")
    torch.npu.synchronize.assert_called_once_with(loader.device)
    assert calls.index("scatter_complete") < calls.index("next_model_collective")
    assert calls[-1] == "next_model_collective"
    assert calls.count("allgather_during_compute") == 1
    assert torch.equal(target[16:-16], source)
    assert torch.all(target[:16] == 0xA5) and torch.all(target[-16:] == 0xA5)
    assert worker.layer_load_finished_events[3].is_set()
    # The join does not consume or drain the following H2D-only lookahead.
    assert requests.get_nowait().pcp_prefetch_only


def test_overlap_layout_is_part_of_rank_plan(monkeypatch):
    worker = make_worker(monkeypatch)
    worker.pcp_load.validate_plan = MagicMock()
    worker.group_block_len = {0: [16]}
    shared = SimpleNamespace(req_ids=["r"], load_keys=["k"], block_ids_arr=np.array([0]), block_gvas_arr=None)
    task = SimpleNamespace(shared_block_data=shared, group_id=0, layer_idx_in_group=0)
    worker.layer_load_tasks = [[task], []]
    worker._validate_pcp_load_plan()
    first = worker.pcp_load.validate_plan.call_args.args[0]
    worker.prefetch_layer_map = {1: (0,)}
    worker._validate_pcp_load_plan()
    assert worker.pcp_load.validate_plan.call_args.args[0] != first
