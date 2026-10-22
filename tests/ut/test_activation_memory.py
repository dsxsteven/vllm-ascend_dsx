# ruff: noqa: SIM117
# Explicit nesting exercises parent/child scope ownership.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.activation_memory import (
    activation_peak_profile_session,
    is_activation_peak_profiling,
    log_ffn_chunk_decision,
    log_mtp_moe_chunk_detail,
    log_serving_peak_request_topk,
    profile_activation,
    profile_runner_phase,
    record_activation_buffer,
    record_activation_peak,
    record_activation_substage,
    record_moe_substage,
    record_mtp_moe_substage,
    serving_memory_log_step,
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
    assert fake_npu.reset_peak_memory_stats.call_count == 5
    # The pinning record is the most-specific scope whose in-scope peak reaches
    # the overall peak. The inner attention scope closed first at peak 170, then
    # the outer target scope closed at peak 180 — so the outer record wins.
    assert profiler.overall_peak_record is not None
    assert profiler.overall_peak_record.category == "target_model"
    assert profiler.overall_peak_record.peak_bytes == 180


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


def test_global_peak_location_locates_pinning_scope():
    # Inner moe scope reaches the absolute peak; outer model scope closes later
    # at the same peak. The pinning record is the most-specific (latest) scope
    # whose in-scope peak equals overall_peak_bytes.
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        # session open:100; target entry ckpt:100; moe entry ckpt:100;
        # moe exit ckpt:200; target exit ckpt:200; session final ckpt:200.
        max_memory_allocated=MagicMock(side_effect=[100, 100, 100, 200, 200, 200]),
        # target baseline:100; moe baseline:130; moe exit after:160; target exit after:170.
        memory_allocated=MagicMock(side_effect=[100, 130, 160, 170]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        activation_peak_profile_session(enabled=True, rank=1) as profiler,
    ):
        with record_activation_peak("target_model", "target.model_forward", num_tokens=64):
            with record_activation_peak(
                "target_moe",
                "target.layer.3.moe",
                layer_idx=3,
                num_tokens=64,
            ):
                pass

    assert profiler is not None
    assert profiler.overall_peak_bytes == 200
    assert profiler.overall_peak_record is not None
    # Equal inherited peaks must never erase the leaf that caused the peak.
    assert profiler.overall_peak_record.category == "target_moe"
    assert profiler.overall_peak_record.peak_bytes == 200

    global_peak_log = next(call for call in mock_warning.call_args_list if "[GLOBAL_PEAK]" in call.args[0])
    line = global_peak_log.args[0] % global_peak_log.args[1:]
    assert "rank=1" in line
    assert "category=TARGET_MOE" in line
    assert "layer=3" in line
    assert "peak_bytes=200" in line
    abs_lines = [c.args[0] % c.args[1:] for c in mock_warning.call_args_list if "[ABS_PEAK]" in c.args[0]]
    assert any("category=TARGET_MODEL" in line for line in abs_lines)
    assert any("category=TARGET_MOE" in line for line in abs_lines)


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


def test_serving_memory_log_step_does_not_sync_or_reset_peak():
    # The serving path must never synchronize or reset the peak tracker (would
    # break ACL graph replay). It only reads bookkeeping values.
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 130, 130]),
        memory_allocated=MagicMock(side_effect=[100, 120, 115]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
    ):
        serving_memory_log_step(rank=2, phase="pre_forward", num_tokens=4096)
        serving_memory_log_step(rank=2, phase="post_forward", num_tokens=4096)

    fake_npu.synchronize.assert_not_called()
    fake_npu.reset_peak_memory_stats.assert_not_called()
    step_logs = [call for call in mock_warning.call_args_list if "[STEP]" in call.args[0]]
    assert len(step_logs) == 2
    pre, post = step_logs
    assert pre.args[1] == 2 and pre.args[2] == "pre_forward" and pre.args[4] == 4096
    assert post.args[1] == 2 and post.args[2] == "post_forward"
    # format string carries the documented fields
    assert pre.args[0] == (
        "[SERVING_MEMORY][STEP] rank=%s phase=%s step=%s tokens=%s "
        "allocated=%.3f GiB allocator_peak_since_reset=%.3f GiB"
    )


class StatefulAllocator:
    """Model real allocator reset semantics, unlike a predetermined value list."""

    def __init__(self):
        self.current = 100
        self.peak = 100

    def synchronize(self):
        pass

    def memory_allocated(self):
        return self.current

    def max_memory_allocated(self):
        return self.peak

    def reset_peak_memory_stats(self):
        self.peak = self.current

    def pulse(self, peak, retained=100):
        self.peak = max(self.peak, peak)
        self.current = retained


def test_unscoped_larger_peak_has_unknown_location():
    allocator = StatefulAllocator()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as log,
    ):
        with activation_peak_profile_session(enabled=True, rank=0) as profiler:
            with record_activation_peak("target_moe", "layer.3.moe", layer_idx=3):
                allocator.pulse(200)
            allocator.pulse(300)
    assert profiler.overall_peak_bytes == 300
    assert profiler.overall_peak_record is None
    line = next(c.args[0] % c.args[1:] for c in log.call_args_list if "[GLOBAL_PEAK]" in c.args[0])
    assert "category=UNKNOWN" in line
    assert "growth_bytes=None" in line


def test_absolute_peak_and_growth_are_selected_from_the_same_instance():
    allocator = StatefulAllocator()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as log,
    ):
        with activation_peak_profile_session(enabled=True, rank=0):
            with record_activation_peak("target_moe", "layer.1", layer_idx=1):
                allocator.pulse(200, retained=190)  # growth 100
            with record_activation_peak("target_moe", "layer.2", layer_idx=2):
                allocator.pulse(220, retained=190)  # largest absolute, growth 30
    line = next(c.args[0] % c.args[1:] for c in log.call_args_list if "[ABS_PEAK]" in c.args[0])
    assert "peak_bytes=220" in line
    assert "baseline_bytes=190" in line
    assert "growth_bytes=30" in line
    assert "layer=2" in line


def test_target_and_mtp_substages_inherit_layer_identity():
    allocator = StatefulAllocator()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.logger.warning"),
    ):
        with activation_peak_profile_session(enabled=True, rank=0) as profiler:
            for phase in ("target", "mtp"):
                with record_activation_peak(f"{phase}_attention", "parent", layer_idx=7):
                    with record_activation_substage("attn_q_proj"):
                        allocator.pulse(200)
                with record_activation_peak(f"{phase}_moe", "parent", layer_idx=7):
                    with record_moe_substage("dispatch", "moe.dispatch"):
                        allocator.pulse(250)
    categories = {r.category for r in profiler.records}
    assert {"target_attn_q_proj", "mtp_attn_q_proj", "target_moe_dispatch", "mtp_moe_dispatch"} <= categories
    assert all(r.layer_idx == 7 for r in profiler.records)
    assert any(r.label == "mtp.layer.7.attn_q_proj" for r in profiler.records)


def test_serving_profiles_merge_forward_and_sampling_and_keep_run_peak():
    allocator = StatefulAllocator()

    class Runner:
        ascend_config = SimpleNamespace(
            activation_peak_debug_serving=True, activation_peak_debug_rank=0, activation_peak_debug_max_steps=2
        )

        @profile_runner_phase("forward")
        def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
            with record_activation_peak("target_moe", "moe", layer_idx=1):
                allocator.pulse(400 if scheduler_output.high else 250)

        @profile_runner_phase("sampling")
        def sample_tokens(self, grammar_output=None):
            with record_activation_substage("head"):
                allocator.pulse(300)

    runner = Runner()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.torch.distributed.is_initialized", return_value=False),
        patch("vllm_ascend.activation_memory.logger.warning"),
    ):
        runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=64, high=True))
        runner.sample_tokens()
        assert runner._activation_peak_step_records.phase == "full_step"
        assert runner._activation_peak_step_records.overall_peak_bytes == 400
        assert runner._activation_peak_step_records.overall_peak_record.category == "target_moe"
        runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=64, high=False))
        runner.sample_tokens()
        assert runner._activation_peak_step_records.overall_peak_bytes == 300
        assert runner._activation_peak_run_result.overall_peak_bytes == 400
        assert not is_activation_peak_profiling()


def test_profile_context_is_cleared_after_checkpoint_failure():
    allocator = StatefulAllocator()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.logger.warning"),
    ):
        try:
            with activation_peak_profile_session(enabled=True, rank=0):
                allocator.synchronize = MagicMock(side_effect=RuntimeError("device error"))
        except RuntimeError:
            pass
    assert not is_activation_peak_profiling()


def test_original_forward_failure_is_not_masked_by_device_cleanup():
    allocator = StatefulAllocator()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.logger.warning"),
        pytest.raises(ValueError, match="original operator error"),
        activation_peak_profile_session(enabled=True, rank=0),
        record_activation_peak("target_moe", "moe"),
    ):
        allocator.synchronize = MagicMock(side_effect=RuntimeError("secondary device error"))
        raise ValueError("original operator error")
    allocator.synchronize.assert_not_called()
    assert not is_activation_peak_profiling()


def test_serving_step_limit_and_dummy_runs_do_not_collect_extra_steps():
    allocator = StatefulAllocator()
    observed = []

    class Runner:
        ascend_config = SimpleNamespace(
            activation_peak_debug_serving=True,
            activation_peak_debug_rank=0,
            activation_peak_debug_max_steps=1,
        )

        @profile_runner_phase("forward")
        def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
            observed.append(is_activation_peak_profiling())
            if is_activation_peak_profiling():
                with record_activation_substage("attn_sfa"):
                    allocator.pulse(200)

    runner = Runner()
    scheduler = SimpleNamespace(total_num_scheduled_tokens=64)
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.torch.distributed.is_initialized", return_value=False),
        patch("vllm_ascend.activation_memory.logger.warning"),
    ):
        runner.execute_model(scheduler, dummy_run=True)
        runner.execute_model(scheduler, is_profile=True)
        runner.execute_model(scheduler)
        runner.execute_model(scheduler)
    assert observed == [False, False, True, False]
    assert runner._activation_peak_step == 2
    assert runner._activation_peak_run_result.overall_peak_bytes == 200


def test_unselected_rank_never_reads_or_resets_allocator():
    fake_npu = MagicMock()

    class Runner:
        ascend_config = SimpleNamespace(
            activation_peak_debug_serving=True,
            activation_peak_debug_rank=0,
            activation_peak_debug_max_steps=0,
        )

        @profile_runner_phase("forward")
        def execute_model(self, scheduler_output):
            assert not is_activation_peak_profiling()
            return "unchanged"

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.torch.distributed.is_initialized", return_value=True),
        patch("vllm_ascend.activation_memory.torch.distributed.get_rank", return_value=1),
    ):
        assert Runner().execute_model(SimpleNamespace(total_num_scheduled_tokens=64)) == "unchanged"
    assert not fake_npu.mock_calls


def test_moe_chunk_labels_preserve_chunk_identity_and_draft_phase():
    allocator = StatefulAllocator()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.logger.warning"),
        activation_peak_profile_session(enabled=True, rank=0) as profiler,
        record_activation_peak("mtp_moe", "mtp.layer.7.moe", layer_idx=7),
    ):
        for index in (1, 2):
            with record_moe_substage("dispatch", f"moe.chunk.{index}.dispatch", num_tokens=32):
                allocator.pulse(200 + index)
    labels = [record.label for record in profiler.records if record.category == "mtp_moe_dispatch"]
    assert labels == ["mtp.layer.7.moe.chunk.1.dispatch", "mtp.layer.7.moe.chunk.2.dispatch"]


def test_precast_internal_router_fast_path_is_measured():
    source = Path(__file__).resolve().parents[2] / "vllm_ascend" / "ops" / "fused_moe" / "fused_moe.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_compute_router_logits")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    allocator = StatefulAllocator()
    result = torch.ones(2, 4)
    linear = MagicMock(side_effect=lambda *args: (allocator.pulse(200), result)[1])
    env = {"torch": torch, "F": SimpleNamespace(linear=linear), "profile_activation": profile_activation}
    exec(compile(module, str(source), "exec"), env)
    gate = MagicMock()
    gate.weight_fp32 = torch.ones(4, 3)
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.logger.warning"),
        activation_peak_profile_session(enabled=True, rank=0) as profiler,
        record_activation_peak("target_moe", "target.layer.4.moe", layer_idx=4),
    ):
        actual = env["_compute_router_logits"](SimpleNamespace(gate=gate), torch.ones(2, 3), torch.ones(2, 3))
    assert actual is result
    gate.assert_not_called()
    linear.assert_called_once()
    assert profiler.overall_peak_record.category == "target_moe_router_logits"
    assert profiler.overall_peak_record.layer_idx == 4


def test_legacy_serving_session_uses_fresh_peak_and_emits_matched_components():
    from vllm_ascend.activation_memory import serving_peak_step_session

    allocator = StatefulAllocator()
    allocator.peak = 900  # A previous operation must not pin this serving step.
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as log,
        serving_peak_step_session(rank=0, num_tokens=64) as profiler,
    ):
        with record_activation_peak("target_model", "target", num_tokens=64):
            with record_activation_substage("attn_q_proj", layer_idx=2):
                allocator.pulse(200)
            with record_activation_substage("moe_combine", layer_idx=2):
                allocator.pulse(300)
    assert profiler.overall_peak_bytes == 300
    assert profiler.most_specific_peak_record().category == "target_moe_combine"
    logs = [call.args[0] % call.args[1:] for call in log.call_args_list]
    assert any("[STEP_PEAK]" in line and "at=MOE_COMBINE" in line for line in logs)
    assert any("[STEP_PEAK_COMP]" in line and "component=MOE_COMBINE" in line for line in logs)


def test_legacy_serving_session_reuses_outer_session_and_clears_after_failure():
    from vllm_ascend.activation_memory import serving_peak_step_session

    allocator = StatefulAllocator()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.logger.warning"),
    ):
        with activation_peak_profile_session(enabled=True, rank=0) as outer:
            with serving_peak_step_session(rank=0, num_tokens=64) as inner:
                assert inner is outer
            assert is_activation_peak_profiling()
        with pytest.raises(ValueError, match="operator"):
            with serving_peak_step_session(rank=0, num_tokens=64):
                raise ValueError("operator")
        assert not is_activation_peak_profiling()


def test_legacy_serving_flag_honors_rank_step_limit_and_sampling():
    allocator = StatefulAllocator()
    active = []

    class Runner:
        ascend_config = SimpleNamespace(
            serving_peak_debug=True,
            serving_peak_debug_rank=1,
            activation_peak_debug_serving=False,
            activation_peak_debug_rank=0,
            activation_peak_debug_max_steps=1,
        )

        @profile_runner_phase("forward")
        def execute_model(self, scheduler_output):
            active.append(is_activation_peak_profiling())
            with record_activation_substage("attn_sfa", layer_idx=4):
                allocator.pulse(200)

        @profile_runner_phase("sampling")
        def sample_tokens(self):
            active.append(is_activation_peak_profiling())
            with record_activation_substage("head"):
                allocator.pulse(300)

    runner = Runner()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.torch.distributed.is_initialized", return_value=True),
        patch("vllm_ascend.activation_memory.torch.distributed.get_rank", return_value=1),
        patch("vllm_ascend.activation_memory.logger.warning"),
    ):
        runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=64))
        runner.sample_tokens()
        assert runner._activation_peak_step_records.phase == "full_step"
        assert runner._activation_peak_step_records.overall_peak_bytes == 300
        runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=64))
        runner.sample_tokens()
    assert active == [True, True, False, False]


def test_equal_sampling_baseline_preserves_known_forward_peak_location():
    allocator = StatefulAllocator()

    class Runner:
        ascend_config = SimpleNamespace(
            activation_peak_debug_serving=True,
            activation_peak_debug_rank=0,
            activation_peak_debug_max_steps=0,
        )

        @profile_runner_phase("forward")
        def execute_model(self, scheduler_output):
            with record_activation_substage("moe_combine", layer_idx=3):
                allocator.pulse(400, retained=400)

        @profile_runner_phase("sampling")
        def sample_tokens(self):
            pass

    runner = Runner()
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.torch.distributed.is_initialized", return_value=False),
        patch("vllm_ascend.activation_memory.logger.warning"),
    ):
        runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=64))
        runner.sample_tokens()
    record = runner._activation_peak_step_records.overall_peak_record
    assert record is not None
    assert record.category == "target_moe_combine"
    assert record.baseline_bytes == 100
    assert record.peak_growth_bytes == 300


def test_serving_peak_request_topk_ranks_by_absolute_peak():
    """End-of-request top-K logs the largest absolute peaks in descending order."""
    allocator = StatefulAllocator()

    class Runner:
        ascend_config = SimpleNamespace(
            activation_peak_debug_serving=True,
            activation_peak_debug_rank=0,
            activation_peak_debug_max_steps=2,
            serving_peak_request_topk=2,
        )

        @profile_runner_phase("forward")
        def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
            with record_activation_substage("attn_sfa", layer_idx=1):
                allocator.pulse(500)
            with record_activation_substage("moe_combine", layer_idx=2):
                allocator.pulse(800)

        @profile_runner_phase("sampling")
        def sample_tokens(self, grammar_output=None):
            with record_activation_substage("head"):
                allocator.pulse(300)

        def _maybe_log_serving_peak_request_topk(self, scheduler_output):
            topk = int(getattr(self.ascend_config, "serving_peak_request_topk", 0))
            if topk <= 0:
                return
            finished = getattr(scheduler_output, "finished_req_ids", None)
            if not finished:
                return
            run_result = getattr(self, "_activation_peak_run_result", None)
            if run_result is None:
                return
            log_serving_peak_request_topk(
                rank=0,
                topk=topk,
                finished_req_ids=finished,
                run_result=run_result,
            )

    runner = Runner()
    finished_ids = {"req-1"}
    with (
        patch("vllm_ascend.activation_memory.torch.npu", allocator, create=True),
        patch("vllm_ascend.activation_memory.torch.distributed.is_initialized", return_value=False),
        patch("vllm_ascend.activation_memory.logger.warning") as log,
    ):
        runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=64))
        runner.sample_tokens()
        runner._maybe_log_serving_peak_request_topk(SimpleNamespace(finished_req_ids=finished_ids))

    # Find the REQ_TOPK_ENTRY lines in emission order; peaks must be descending.
    entries = [
        args[0]
        for (args, _kwargs) in log.call_args_list
        if args and "[SERVING_MEMORY][REQ_TOPK_ENTRY]" in str(args[0])
    ]
    assert len(entries) == 2
    assert "MOE_COMBINE" in entries[0]
    assert "ATTN_SFA" in entries[1]


def test_serving_peak_request_topk_noop_when_disabled():
    """topk<=0 or missing run_result is a silent no-op."""
    with patch("vllm_ascend.activation_memory.logger.warning") as log:
        log_serving_peak_request_topk(
            rank=0, topk=0, finished_req_ids={"req-1"}, run_result=None
        )
        log_serving_peak_request_topk(
            rank=0, topk=5, finished_req_ids={"req-1"}, run_result=None
        )
    log.assert_not_called()

