# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore completion must precede communication, not local CPU submission."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence as fence
from vllm_ascend.ops.fused_moe import moe_comm_method as moe


@pytest.fixture
def pending(monkeypatch):
    monkeypatch.setattr(fence, "_attention_compute_start_gate", None)
    monkeypatch.setattr(torch.npu, "Event", MagicMock())
    monkeypatch.setattr(torch.npu, "current_stream", MagicMock())
    calls = []
    gate = fence.reset_attention_compute_start_gate()
    gate.on_overlap_start = lambda: calls.append("start")
    gate.on_overlap_finish = lambda: calls.append("join")
    return gate, calls


def submit_local(calls):
    with fence.indexer_attention_transfer_window(defer_join=True):
        fence.start_local_indexer_transfer_overlap()
        calls.append("LI")
        with fence.attention_transfer_window(allow_pcp_overlap=True):
            calls.append("SFA")
    calls.extend(["V-up", "O-proj", "RMSNorm", "router"])


def test_local_submission_precedes_join_and_next_layer_is_fenced(pending):
    gate, calls = pending
    submit_local(calls)
    assert calls == ["start", "LI", "SFA", "V-up", "O-proj", "RMSNorm", "router"]
    fence.reset_attention_compute_start_gate()
    assert calls[-1] == "join"
    gate.finish_overlap()
    assert calls.count("join") == 1


def test_exception_still_drains_deferred_restore(pending):
    _, calls = pending
    with pytest.raises(ValueError), fence.indexer_attention_transfer_window(defer_join=True):
        fence.start_local_indexer_transfer_overlap()
        raise ValueError("LI failed")
    assert calls == ["start", "join"]


def test_legacy_finish_hook_keeps_original_boundary(pending):
    gate, calls = pending
    gate.on_finish = lambda: calls.append("legacy_finish")
    submit_local(calls)
    assert calls.index("join") < calls.index("legacy_finish") < calls.index("V-up")


@pytest.mark.parametrize("local", [False, True])
def test_moe_prepare_joins_before_any_communication(monkeypatch, pending, local):
    _, calls = pending
    submit_local(calls)
    if local:
        prepare = object.__new__(moe.PrepareAndFinalizeWithMC2)
        monkeypatch.setattr(moe.PrepareAndFinalizeWithMC2, "prepare", lambda *a, **kw: calls.append("local_prepare"))
    else:
        prepare = SimpleNamespace(prepare=lambda **kw: calls.append("communicating_prepare"))
    moe.MoECommMethod.prepare(SimpleNamespace(prepare_finalize=prepare), torch.empty(1), torch.empty(1))
    if local:
        assert "join" not in calls
    else:
        assert calls[-2:] == ["join", "communicating_prepare"]
    fence.finish_attention_transfer_overlap()


def test_fused_mc2_weight_preparation_then_join_before_launch(monkeypatch, pending):
    _, calls = pending
    submit_local(calls)
    monkeypatch.setattr(moe, "_EXTRA_CTX", SimpleNamespace(use_mega_moe=True, is_decode_only_node=False))
    method = object.__new__(moe.FusedMC2CommImpl)
    method.token_dispatcher = MagicMock(spec=moe.TokenDispatcherWithMC2)
    method.enable_fused_mc2 = 1
    method._apply_cann_mega_moe = lambda *a, **kw: (calls.append("MegaMoE"), None)
    quant = SimpleNamespace(get_fused_mc2_weights=lambda layer: calls.append("prepare_weights"))
    method.fused_experts(SimpleNamespace(layer=None), quant)
    assert calls[-3:] == ["prepare_weights", "join", "MegaMoE"]
