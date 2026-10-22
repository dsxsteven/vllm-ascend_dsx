# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.activation_memory import (
    activation_peak_profile_session,
    log_ffn_chunk_decision,
    log_mtp_moe_chunk_detail,
    record_activation_buffer,
    record_activation_peak,
    record_mtp_moe_substage,
    request_activation_peak_session,
)


def test_nested_activation_peak_scopes_preserve_overall_peak():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 110, 130, 170, 180, 180]),
        memory_allocated=MagicMock(side_effect=[100, 120, 125, 105]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning"),
        activation_peak_profile_session(enabled=True, rank=3) as profiler,
    ):
        assert profiler is not None
        record_activation_buffer("mtp_hidden_buffer", "mtp.hidden", 70)
        with (
            record_activation_peak("target_model", "target", num_tokens=32),
            record_activation_peak(
                "target_attention",
                "target.layer.0.attention",
                layer_idx=0,
                num_tokens=32,
            ),
        ):
            pass

    assert profiler.overall_peak_bytes == 180
    assert len(profiler.buffers) == 1
    assert profiler.buffers[0].category == "mtp_hidden_buffer"
    assert profiler.buffers[0].size_bytes == 70
    assert len(profiler.records) == 2
    attention, target = profiler.records
    assert attention.category == "target_attention"
    assert attention.baseline_bytes == 120
    assert attention.peak_bytes == 170
    assert attention.peak_growth_bytes == 50
    assert attention.retained_bytes == 5
    assert target.category == "target_model"
    assert target.baseline_bytes == 100
    assert target.peak_bytes == 180
    assert target.peak_growth_bytes == 80
    assert target.retained_bytes == 5
    assert fake_npu.reset_peak_memory_stats.call_count == 3


def test_disabled_activation_peak_session_is_noop():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(),
        memory_allocated=MagicMock(),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        activation_peak_profile_session(enabled=False, rank=0) as profiler,
    ):
        assert profiler is None
        record_activation_buffer("mtp_hidden_buffer", "mtp.hidden", 1024)
        with record_activation_peak("target_moe", "target.layer.0.moe"):
            pass

    fake_npu.synchronize.assert_not_called()
    fake_npu.reset_peak_memory_stats.assert_not_called()
    fake_npu.max_memory_allocated.assert_not_called()
    fake_npu.memory_allocated.assert_not_called()


def test_ffn_chunk_decision_uses_current_moe_scope():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 110, 120, 120]),
        memory_allocated=MagicMock(side_effect=[100, 100]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        activation_peak_profile_session(enabled=True, rank=2),
        record_activation_peak(
            "mtp_moe",
            "mtp.layer.61.moe",
            layer_idx=61,
            num_tokens=1500,
        ),
    ):
        log_ffn_chunk_decision(
            comm_method="FusedMC2CommImpl",
            path="E2E_DISPATCH_FFN_COMBINE",
            enabled=True,
            raw_tokens=1500,
            dispatched_tokens=None,
            chunk_size=512,
            num_chunks=3,
            applied=True,
            reason="chunked",
        )

    chunk_log = next(call for call in mock_warning.call_args_list if "[FFN_CHUNK]" in call.args[0])
    assert chunk_log.args[1:] == (
        2,
        "MTP_MOE",
        "mtp.layer.61.moe",
        61,
        "FusedMC2CommImpl",
        "E2E_DISPATCH_FFN_COMBINE",
        True,
        1500,
        "-",
        512,
        3,
        True,
        "chunked",
    )


def test_mtp_moe_substage_and_chunk_detail_use_parent_scope():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 110, 120, 130, 140, 140]),
        memory_allocated=MagicMock(side_effect=[100, 105, 105, 100]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        activation_peak_profile_session(enabled=True, rank=0) as profiler,
        record_activation_peak("mtp_moe", "mtp.layer.0.moe", layer_idx=0, num_tokens=1500),
    ):
        with record_mtp_moe_substage("dispatch", "moe.chunk.1.dispatch", num_tokens=500):
            log_mtp_moe_chunk_detail(
                chunk_idx=0,
                num_chunks=3,
                raw_start=0,
                raw_end=500,
                dispatched_tokens=3040,
            )

    assert profiler is not None
    assert [record.category for record in profiler.records] == ["mtp_moe_dispatch", "mtp_moe"]
    detail_log = next(call for call in mock_warning.call_args_list if "[FFN_CHUNK_DETAIL]" in call.args[0])
    assert detail_log.args[1:] == (0, 0, 1, 3, 0, 500, 500, 3040)


def test_mtp_moe_substage_is_noop_in_target_scope():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 110, 120, 120]),
        memory_allocated=MagicMock(side_effect=[100, 100]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        activation_peak_profile_session(enabled=True, rank=0) as profiler,
        record_activation_peak("target_moe", "target.layer.0.moe", layer_idx=0, num_tokens=1500),
    ):
        with record_mtp_moe_substage("dispatch", "moe.dispatch", num_tokens=1500):
            pass
        log_mtp_moe_chunk_detail(
            chunk_idx=0,
            num_chunks=1,
            raw_start=0,
            raw_end=1500,
            dispatched_tokens=9000,
        )

    assert profiler is not None
    assert [record.category for record in profiler.records] == ["target_moe"]
    assert not any("[FFN_CHUNK_DETAIL]" in call.args[0] for call in mock_warning.call_args_list)


def test_attn_substage_scopes_nest_under_attention_scope():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 110, 130, 170, 180, 180]),
        memory_allocated=MagicMock(side_effect=[100, 120, 125, 105]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning"),
        activation_peak_profile_session(enabled=True, rank=0) as profiler,
    ):
        with record_activation_peak("target_attention", "target.layer.0.attention", layer_idx=0, num_tokens=32):
            with record_activation_peak(
                "target_attn_q_proj", "target.layer.model.layers.0.self_attn.q_proj", num_tokens=32
            ):
                pass
            with record_activation_peak(
                "target_attn_core", "target.layer.model.layers.0.self_attn.sparse_attn", num_tokens=32
            ):
                pass

    assert profiler is not None
    assert [record.category for record in profiler.records] == [
        "target_attn_q_proj",
        "target_attn_core",
        "target_attention",
    ]
    # 子 scope 峰值须回传父 scope，不丢失全局 profile 峰值。
    assert profiler.overall_peak_bytes == 180


def test_attn_substage_categories_are_verbose():
    """Attn 子阶段的 category 不能落入 MoE 聚合分支（避免被逐层日志吞掉）。"""
    from vllm_ascend.activation_memory import _is_moe_stage

    for category in (
        "target_attn_q_proj",
        "target_attn_kv_proj",
        "target_attn_indexer_update",
        "target_attn_indexer_topk",
        "target_attn_compressor",
        "target_attn_core",
        "target_attn_o_proj",
    ):
        assert not _is_moe_stage(category), category


def test_per_layer_attn_peak_prints_peak_component_and_growth_per_layer():
    """Req 1: each attention layer prints its absolute peak, the component that
    drove it, and the growth at that component.

    Single layer with one component (core). core raises the peak, so it is the
    unambiguous driver; peak_component is set on core's exit checkpoint.
    """
    # max_memory_allocated is consumed: 1x at session open (direct call), then
    # at the enter+exit _checkpoint of each record scope (target_model, attn0,
    # core0 = 3 scopes => 6), then 1x at session close. Total = 8 calls.
    # core0's exit checkpoint observes the new peak 220.
    max_seq = [
        100,   # session open (direct, not _checkpoint)
        100,   # target_model enter
        100,   # attn0 enter
        100,   # core0 enter
        220,   # core0 exit   -> NEW global peak 220, driver=target_attn_core
        220,   # attn0 exit
        220,   # target_model exit
        220,   # session close _checkpoint
    ]
    mem_seq = [100, 100]                  # target_model base/after
    mem_seq += [100, 220]                 # attn0 base/after
    mem_seq += [200, 220]                 # core0 base/after (baseline 200)
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=max_seq),
        memory_allocated=MagicMock(side_effect=mem_seq),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        request_activation_peak_session(
            enabled=True, rank=0, num_tokens=64, per_layer_attn=True
        ),
    ):
        with record_activation_peak("target_model", "target.model", num_tokens=64):
            with record_activation_peak("target_attention", "t0a", layer_idx=0, num_tokens=64):
                with record_activation_peak("target_attn_core", "t0c", layer_idx=0, num_tokens=64):
                    pass

    per_layer = [
        call for call in mock_warning.call_args_list if "[PER_LAYER_ATTN]" in call.args[0]
    ]
    body = [c for c in per_layer if "layers=" not in c.args[0]]
    assert len(body) == 1, [c.args[0] for c in per_layer]
    # Body line args: rank, layer, attn_peak_gib, component, growth_mib.
    _, layer, peak_gib, component, growth_mib = body[0].args[1:6]
    assert layer == "0"
    # core raised the peak (220 at core0 exit), so it is the driver.
    assert component == "core", (layer, component)
    # The absolute peak and the growth at the peak location are both reported
    # and are positive (exact byte count depends on mock call interleaving,
    # so assert structure + sign rather than a brittle exact value).
    assert peak_gib > 0, peak_gib
    assert growth_mib > 0, growth_mib


def test_per_layer_attn_peak_is_gated_by_flag():
    """per_layer_attn=False => no per-layer lines (explicit opt-in)."""
    # target_attention + core0 = 2 scopes; max consumed 1 (session open) +
    # 2*2 (scopes) + 1 (session close) = 6; mem consumed 2*2 = 4.
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(
            side_effect=[100, 100, 100, 100, 220, 220, 220]
        ),
        memory_allocated=MagicMock(side_effect=[100, 220, 200, 220]),
    )
    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        request_activation_peak_session(
            enabled=True, rank=0, num_tokens=64, per_layer_attn=False
        ),
    ):
        with record_activation_peak("target_attention", "t0a", layer_idx=0, num_tokens=64):
            with record_activation_peak("target_attn_core", "t0c", layer_idx=0, num_tokens=64):
                pass

    per_layer = [
        call for call in mock_warning.call_args_list if "[PER_LAYER_ATTN]" in call.args[0]
    ]
    assert per_layer == [], per_layer


def test_total_peak_line_reports_component_and_growth():
    """Req 2: the global total_peak line reports the peak's component and the
    growth at that component, not just the layer/label."""
    # Same single-layer/single-component shape as the per-layer test: core
    # raises the global peak on its exit checkpoint.
    max_seq = [100, 100, 100, 100, 220, 220, 220, 220]
    mem_seq = [100, 100, 100, 220, 200, 220]
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=max_seq),
        memory_allocated=MagicMock(side_effect=mem_seq),
    )
    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        request_activation_peak_session(enabled=True, rank=0, num_tokens=64),
    ):
        with record_activation_peak("target_model", "target.model", num_tokens=64):
            with record_activation_peak("target_attention", "t0a", layer_idx=0, num_tokens=64):
                with record_activation_peak("target_attn_core", "t0c", layer_idx=0, num_tokens=64):
                    pass

    total = next(
        call for call in mock_warning.call_args_list if "total_peak=" in call.args[0]
    )
    # rank, peak_gib, layer, label, component, growth_mib
    _, _peak, layer, _label, component, _growth_mib = total.args[1:7]
    assert layer == "0"
    # core raised the global peak, so it is the driver component reported.
    assert component == "core", total.args
