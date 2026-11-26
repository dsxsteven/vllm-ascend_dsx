# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the real indexer path and bounded LI/SFA restore lifecycle."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.attention import indexer as indexer_module
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence as fence


@pytest.fixture
def events(monkeypatch):
    monkeypatch.setattr(fence, "_attention_compute_start_gate", None)
    monkeypatch.setattr(torch.npu, "Event", MagicMock())
    monkeypatch.setattr(torch.npu, "current_stream", MagicMock())
    events = []
    gate = fence.reset_attention_compute_start_gate()
    gate.on_overlap_start = lambda: events.append("restore_start")
    gate.on_overlap_finish = lambda: events.append("restore_join")
    gate.on_start = lambda: events.append("save_gate")
    gate.on_finish = lambda: events.append("save_finish")
    return events


def make_indexer(monkeypatch, events, tokens):
    model = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(model)
    model._dcp_size = 1
    model._sharded_dcp = False
    model.head_dim = 2
    model.topk_tokens = 2
    model.enable_sparse_li_c8 = model.enable_sparse_li_c4 = False
    model.use_torch_npu_lightning_indexer = True
    hidden = torch.arange(tokens * 2, dtype=torch.float32).reshape(tokens, 2)
    keys, weights = hidden.clone(), torch.ones(tokens, 1)
    cache = torch.zeros_like(keys)
    model.k_cache = SimpleNamespace(kv_cache=(cache,))
    model.wq_b = model.wk_weights_proj = MagicMock()
    model._select_execution_paths = lambda *args: (None, False, False)
    model.forward_k = lambda *args: (keys, None, weights)

    def gather(*args):
        events.append("cache_gather")
        return keys, None, torch.arange(tokens)

    def write(*args):
        events.append("cache_write")
        cache.copy_(keys)

    def project(*args):
        events.append("query_project")
        return hidden[:, None], None, (tokens, 1, 2)

    def select(*args, **kwargs):
        events.append("local_li")
        torch.testing.assert_close(args[4][0], keys)
        return args[0].sum(-1).to(torch.int32)

    model._gather_cache_inputs = gather
    model.write_cache = write
    model._forward_q = project
    monkeypatch.setattr(indexer_module.DeviceOperator, "indexer_select_post_process", select)
    metadata = SimpleNamespace(
        cos=None,
        sin=None,
        cache_scatter_indices=None,
        num_actual_tokens=tokens,
        actual_seq_lengths_query=torch.tensor([tokens]),
        actual_seq_lengths_key=torch.tensor([tokens]),
        kv_gather_block_ids=None,
        topk_selector=None,
    )
    return model, hidden, metadata


@pytest.mark.parametrize("tokens,compute_topk", [(4, True), (0, True), (0, False), (4, False)])
@pytest.mark.parametrize("enabled", [False, True])
def test_local_indexer_and_sfa_share_one_restore(monkeypatch, events, tokens, compute_topk, enabled):
    model, hidden, metadata = make_indexer(monkeypatch, events, tokens)
    gate = fence.get_attention_compute_start_gate()
    if not enabled:
        gate.on_overlap_start = gate.on_overlap_finish = None
    with fence.indexer_attention_transfer_window():
        result = model.forward(hidden, hidden, hidden, metadata, compute_topk=compute_topk)
        assert "save_gate" not in events
        assert "restore_join" not in events
        with fence.attention_transfer_window(allow_pcp_overlap=True):
            events.append("sfa")
    events.append("model_collective")
    assert events[:2] == ["cache_gather", "cache_write"]
    assert events.count("restore_start") == events.count("restore_join") == int(enabled)
    assert events.index("save_gate") < events.index("sfa") < events.index("save_finish")
    if tokens and compute_topk:
        torch.testing.assert_close(result, hidden[:, None].sum(-1).to(torch.int32))
        assert events.index("query_project") < events.index("local_li") < events.index("sfa")
    else:
        assert result is None and "local_li" not in events
    if enabled:
        assert events.index("restore_start") < events.index("save_gate")
        if "local_li" in events:
            assert events.index("restore_start") < events.index("local_li")
        assert events.index("sfa") < events.index("restore_join") < events.index("model_collective")


@pytest.mark.parametrize("fallback", ["dcp", "sharded", "cache_gather", "custom_selector"])
@pytest.mark.parametrize("tokens", [0, 4])
def test_communication_paths_keep_sfa_boundary(monkeypatch, events, fallback, tokens):
    model, _, metadata = make_indexer(monkeypatch, events, tokens)
    if fallback == "dcp":
        model._dcp_size = 2
    elif fallback == "sharded":
        model._sharded_dcp = True
    elif fallback == "cache_gather":
        metadata.kv_gather_block_ids = torch.tensor([0])
    else:
        metadata.topk_selector = lambda *args: None
    with fence.indexer_attention_transfer_window():
        model._start_local_topk_overlap(metadata)
        assert events == []
        with fence.attention_transfer_window(allow_pcp_overlap=True):
            events.append("sfa")
    assert events == ["save_gate", "restore_start", "sfa", "restore_join", "save_finish"]


@pytest.mark.parametrize("failure", ["start", "li", "sfa", "join"])
def test_exceptions_close_restore_once(events, failure):
    gate = fence.get_attention_compute_start_gate()

    def fail(stage):
        events.append(stage)
        raise RuntimeError(stage)

    if failure == "start":
        gate.on_overlap_start = lambda: fail("restore_start")
    if failure == "join":
        gate.on_overlap_finish = lambda: fail("restore_join")
    with pytest.raises(RuntimeError), fence.indexer_attention_transfer_window():
        fence.start_local_indexer_transfer_overlap()
        if failure == "li":
            fail("local_li")
        with fence.attention_transfer_window(allow_pcp_overlap=True):
            if failure == "sfa":
                fail("sfa")
    assert gate.indexer_scope_depth == 0
    assert events.count("restore_start") == events.count("restore_join") == 1
    assert events.count("save_gate") == int(failure in ("sfa", "join"))


def test_standalone_indexer_cannot_launch_restore(monkeypatch, events):
    model, hidden, metadata = make_indexer(monkeypatch, events, 4)
    model.forward(hidden, hidden, hidden, metadata)
    assert events == ["cache_gather", "cache_write", "query_project", "local_li"]


def test_scope_without_indexer_uses_sfa_fallback(events):
    with fence.indexer_attention_transfer_window():
        assert events == []
        with fence.attention_transfer_window(allow_pcp_overlap=True):
            events.append("sfa")
    assert events == ["save_gate", "restore_start", "sfa", "restore_join", "save_finish"]


def test_cancelled_scope_cannot_launch_restore(events):
    fence.get_attention_compute_start_gate().cancel()
    with fence.indexer_attention_transfer_window():
        fence.start_local_indexer_transfer_overlap()
        with fence.attention_transfer_window(allow_pcp_overlap=True):
            pass
    assert events == []
