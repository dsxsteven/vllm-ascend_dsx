# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scoped activation-memory peak diagnostics for Ascend profile runs.

Usage
-----
Set in ``--additional-config`` (requires ``enforce_eager``)::

    "activation_peak_debug": true,          # master switch
    "activation_peak_debug_rank": 0         # 0 = rank 0 only, -1 = all ranks

During ``profile_run`` the runner wraps the whole forward in
``activation_peak_profile_session``; every ``record_activation_peak`` scope in
the model/runner path then measures its own transient peak. Each scope entry
takes a checkpoint (synchronize + propagate current peak to parent scopes),
records ``memory_allocated`` as baseline, resets the allocator peak tracker,
and on exit reads the in-scope peak — so nested scopes never lose their peak
to a child's reset and every parent's peak bounds its children's.

Scope coverage (full forward)
----------------------------
- ``target_model``          : whole model forward (KV-pricing denominator)
- ``target_embedding``      : embed_input_ids
- ``target_hc_expand``      : hc repeat-expand to (N, hc_mult, H)
- ``target_layer``          : per-layer total (− attention − moe = unattributed)
- ``target_attention``      : per-layer self_attn (+SP gather/scatter)
- ``target_attn_qkv_proj``  : attn q/kv projections (wq_a/q_norm/wq_b/wkv/kv_norm/RoPE;
  MLA: q_a/q_b/kv_a/kv_b preprocess; GLM-5.2/MLA path)
- ``target_attn_indexer``   : attn indexer query/weights + quantize (DSA V41 only)
- ``target_attn_compressor``: attn compressor write (compressed KV source; DSA V41 only)
- ``target_attn_sfa``       : attn SparseFlashMla / MLA decode+prefill core
- ``target_attn_o_proj``    : attn output projection (wo_a/wo_b, incl. OTP)
- ``target_moe``            : per-layer mlp (router+shared+routed)
- ``target_sp_gather``      : tail sp_all_gather back to full tokens
- ``target_tail``           : hc_head + final norm
- ``target_head``           : lm_head (dummy sampler)
- ``mtp_*``                 : draft-side scopes when speculative decoding is on

Reading the logs
----------------
``[ACTIVATION_PEAK][CONFIG]``     session parameters (ffn_chunking, chunk_size …)
``[ACTIVATION_PEAK][STAGE]``      per scope instance:
    growth  = in-scope peak − entry baseline  (reducible transient)
    retained= after − baseline  (positive ⇒ leak / persistent allocation)
``[ACTIVATION_PEAK][FFN_CHUNK]``  per-MoE-layer chunk decision (applied/reason)
``[ACTIVATION_PEAK][SUMMARY]``   per category: max growth over all instances —
    the share of ``target_model.max_growth`` is "who pins the pricing peak".
``[ACTIVATION_PEAK][GLOBAL_PEAK]`` the single scope instance whose in-scope peak
    equals ``overall_peak_bytes`` — i.e. *where* the absolute memory peak sits
    (category/label/layer/tokens) plus its growth and entry baseline.
``[ACTIVATION_PEAK][VERDICT]``   dominant top-level vs dominant component.

Caveats
-------
- Counts only torch-allocator tensors (HCCL/CANN workspaces excluded).
- Two synchronizes per scope: debug-only; do not use numbers for perf timing.
- Scopes reset the global peak tracker; NPUWorker compensates KV sizing via
  ``activation_peak_profile_result`` — keep that assignment if you move the
  session wrapper.

Serving-path memory logging
---------------------------
Separate from the profile_run scope machinery above, ``serving_memory_log_step``
prints a per-step allocator snapshot during real request serving (not just
profile_run). It is intentionally lightweight:

- No ``torch.npu.synchronize()``, no ``reset_peak_memory_stats`` — safe under
  ACL graph capture/replay (the profile scopes would break graph replay).
- Reads only ``memory_allocated()`` / ``max_memory_allocated()``, which are
  bookkeeping reads from the allocator and do not force a device sync. Values
  may lag in-flight allocations by one step; that is acceptable for a serving
  occupancy log.
- Gated by the same ``activation_peak_debug`` config flag, so enabling the flag
  turns on BOTH profile_run scopes and serving-path logging. When the flag is
  off, ``serving_memory_log_step`` is a no-op (callers should still gate the
  call on the cached flag to avoid the function-call overhead per step).
- ``max_memory_allocated`` is never reset on the serving path, so the reported
  ``run_peak`` is the cumulative high-water mark since process start (the
  allocator peak is only reset inside ``activation_peak_profile_session``).

Per-step scoped serving-peak mode
---------------------------------
``serving_peak_debug`` (AscendConfig, requires ``enforce_eager``) is a heavier
debug mode that wraps *each* serving forward in a throwaway
``ActivationPeakProfiler`` via ``serving_peak_step_session``. The existing
``record_activation_peak`` scopes in the model/attention/MoE paths then fire
for that one forward, and on exit ``_log_serving_peak_step`` prints a
``[SERVING_MEMORY][STEP_PEAK]`` block with:

- the step's absolute memory peak and the scope that pins it (moe / attn_sfa /
  attn_qkv_proj / attn_o_proj / sp_gather / tail / head …) — the most-specific
  scope whose in-scope high-water mark reaches the absolute peak;
- each attention sub-component's absolute peak plus its growth delta.

This mode costs two ``torch.npu.synchronize()`` calls per scope per step and
resets the allocator peak tracker per step, so it cannot coexist with the
lightweight ``[SERVING_MEMORY][STEP]`` log (whose cumulative ``run_peak`` would
be clobbered by the per-step reset). Callers enable one mode or the other; when
``serving_peak_debug`` is on, the runner skips the lightweight STEP log. Do not
use under ACL graph capture/replay — ``enforce_eager`` is enforced in AscendConfig.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from vllm.logger import logger

_MIB = 1024**2
_GIB = 1024**3


@dataclass(frozen=True)
class ActivationPeakRecord:
    category: str
    label: str
    layer_idx: int | None
    num_tokens: int | None
    baseline_bytes: int
    peak_bytes: int
    allocated_after_bytes: int
    peak_growth_bytes: int
    retained_bytes: int
    elapsed_ms: float


@dataclass(frozen=True)
class ActivationBufferRecord:
    category: str
    label: str
    size_bytes: int


@dataclass
class _ActiveScope:
    category: str
    label: str
    layer_idx: int | None
    num_tokens: int | None
    baseline_bytes: int
    started_at: float
    peak_bytes: int = 0


@dataclass
class ActivationPeakProfiler:
    """Collect nested peak measurements without losing the global profile peak."""

    rank: int
    records: list[ActivationPeakRecord] = field(default_factory=list)
    buffers: list[ActivationBufferRecord] = field(default_factory=list)
    overall_peak_bytes: int = 0
    # The record whose in-scope peak equals overall_peak_bytes — i.e. the
    # scope instance that *pins* the absolute memory peak. Updated on scope
    # exit whenever a freshly-closed record reaches the running overall peak.
    overall_peak_record: ActivationPeakRecord | None = None
    _scopes: list[_ActiveScope] = field(default_factory=list)

    def most_specific_peak_record(self) -> ActivationPeakRecord | None:
        """Return the most-specific scope whose in-scope peak equals the
        absolute high-water mark.

        ``overall_peak_record`` is updated on scope exit with a last-wins tie
        rule, so when an inner scope and its outer parent close at the same
        peak the (less specific) parent wins. For locating *where* the peak
        sits we want the most specific (innermost) scope that reached the
        peak. Records are appended on scope exit in close order (inner scopes
        close before their parents), so the first record whose ``peak_bytes``
        equals ``overall_peak_bytes`` is the innermost scope that pinned the
        peak. Falls back to ``overall_peak_record`` and then to the raw
        max-peak record when no record matches exactly.
        """
        return _most_specific_peak_record(self)

    def _synchronize(self) -> None:
        torch.npu.synchronize()

    def _checkpoint(self) -> int:
        self._synchronize()
        peak_bytes = int(torch.npu.max_memory_allocated())
        self.overall_peak_bytes = max(self.overall_peak_bytes, peak_bytes)
        for scope in self._scopes:
            scope.peak_bytes = max(scope.peak_bytes, peak_bytes)
        return peak_bytes

    @contextmanager
    def record(
        self,
        category: str,
        label: str,
        *,
        layer_idx: int | None = None,
        num_tokens: int | None = None,
    ) -> Iterator[None]:
        # Preserve parent work performed before resetting the peak for this
        # child. Nested scopes propagate their peak back to every parent.
        self._checkpoint()
        baseline_bytes = int(torch.npu.memory_allocated())
        torch.npu.reset_peak_memory_stats()
        scope = _ActiveScope(
            category=category,
            label=label,
            layer_idx=layer_idx,
            num_tokens=num_tokens,
            baseline_bytes=baseline_bytes,
            started_at=time.perf_counter(),
            peak_bytes=baseline_bytes,
        )
        self._scopes.append(scope)
        try:
            yield
        finally:
            self._checkpoint()
            allocated_after_bytes = int(torch.npu.memory_allocated())
            self._scopes.pop()
            peak_bytes = max(scope.peak_bytes, allocated_after_bytes)
            if self._scopes:
                self._scopes[-1].peak_bytes = max(self._scopes[-1].peak_bytes, peak_bytes)
            record = ActivationPeakRecord(
                category=category,
                label=label,
                layer_idx=layer_idx,
                num_tokens=num_tokens,
                baseline_bytes=baseline_bytes,
                peak_bytes=peak_bytes,
                allocated_after_bytes=allocated_after_bytes,
                peak_growth_bytes=max(0, peak_bytes - baseline_bytes),
                retained_bytes=allocated_after_bytes - baseline_bytes,
                elapsed_ms=(time.perf_counter() - scope.started_at) * 1000,
            )
            self.records.append(record)
            # A scope whose in-scope peak matches the running absolute peak is
            # the instance that pins the global memory high-water mark. The
            # last such record wins (ties are expected for outer scopes whose
            # peak is set by a child); keep the most specific (latest closed).
            if peak_bytes >= self.overall_peak_bytes:
                self.overall_peak_record = record
            logger.warning(
                "[ACTIVATION_PEAK][STAGE] rank=%s category=%s label=%s "
                "layer=%s tokens=%s baseline=%.3f GiB peak=%.3f GiB "
                "growth=%.3f MiB retained=%+.3f MiB elapsed=%.3f ms",
                self.rank,
                category.upper(),
                label,
                "-" if layer_idx is None else layer_idx,
                "-" if num_tokens is None else num_tokens,
                baseline_bytes / _GIB,
                peak_bytes / _GIB,
                record.peak_growth_bytes / _MIB,
                record.retained_bytes / _MIB,
                record.elapsed_ms,
            )

    def record_buffer(self, category: str, label: str, size_bytes: int) -> None:
        record = ActivationBufferRecord(
            category=category,
            label=label,
            size_bytes=size_bytes,
        )
        self.buffers.append(record)
        logger.warning(
            "[ACTIVATION_PEAK][BUFFER] rank=%s category=%s label=%s persistent=%.3f MiB (%.3f GiB)",
            self.rank,
            category.upper(),
            label,
            size_bytes / _MIB,
            size_bytes / _GIB,
        )

    def log_summary(self) -> None:
        if not self.records and not self.buffers:
            logger.warning("[ACTIVATION_PEAK][SUMMARY] rank=%s no measured stages", self.rank)
            return

        category_winners: dict[str, ActivationPeakRecord] = {}
        absolute_peak_winners: dict[str, ActivationPeakRecord] = {}
        for record in self.records:
            previous = category_winners.get(record.category)
            if previous is None or record.peak_growth_bytes > previous.peak_growth_bytes:
                category_winners[record.category] = record
            prev_abs = absolute_peak_winners.get(record.category)
            if prev_abs is None or record.peak_bytes > prev_abs.peak_bytes:
                absolute_peak_winners[record.category] = record

        for category, record in sorted(category_winners.items()):
            logger.warning(
                "[ACTIVATION_PEAK][SUMMARY] rank=%s category=%s max_growth=%.3f MiB "
                "absolute_peak=%.3f GiB label=%s layer=%s tokens=%s",
                self.rank,
                category.upper(),
                record.peak_growth_bytes / _MIB,
                record.peak_bytes / _GIB,
                record.label,
                "-" if record.layer_idx is None else record.layer_idx,
                "-" if record.num_tokens is None else record.num_tokens,
            )

        # Where each category's absolute high-water mark sits — useful for
        # locating the memory peak independently of transient growth.
        for category, record in sorted(absolute_peak_winners.items()):
            logger.warning(
                "[ACTIVATION_PEAK][ABS_PEAK] rank=%s category=%s absolute_peak=%.3f GiB "
                "growth_at_peak=%.3f MiB baseline=%.3f GiB label=%s layer=%s tokens=%s",
                self.rank,
                category.upper(),
                record.peak_bytes / _GIB,
                record.peak_growth_bytes / _MIB,
                record.baseline_bytes / _GIB,
                record.label,
                "-" if record.layer_idx is None else record.layer_idx,
                "-" if record.num_tokens is None else record.num_tokens,
            )

        buffer_winners: dict[str, ActivationBufferRecord] = {}
        for record in self.buffers:
            previous = buffer_winners.get(record.category)
            if previous is None or record.size_bytes > previous.size_bytes:
                buffer_winners[record.category] = record
        for category, record in sorted(buffer_winners.items()):
            logger.warning(
                "[ACTIVATION_PEAK][SUMMARY] rank=%s category=%s persistent=%.3f MiB (%.3f GiB) label=%s",
                self.rank,
                category.upper(),
                record.size_bytes / _MIB,
                record.size_bytes / _GIB,
                record.label,
            )

        top_level = [
            record
            for category in ("target_model", "mtp_draft")
            if (record := category_winners.get(category)) is not None
        ]
        components = [
            record for category, record in category_winners.items() if category not in {"target_model", "mtp_draft"}
        ]
        dominant_top = max(top_level, key=lambda item: item.peak_growth_bytes, default=None)
        dominant_activation = max(components, key=lambda item: item.peak_growth_bytes, default=None)
        dominant_buffer = max(buffer_winners.values(), key=lambda item: item.size_bytes, default=None)
        if dominant_buffer is not None and (
            dominant_activation is None or dominant_buffer.size_bytes > dominant_activation.peak_growth_bytes
        ):
            dominant_component_category = dominant_buffer.category.upper()
            dominant_component_kind = "PERSISTENT_BUFFER"
            dominant_component_bytes = dominant_buffer.size_bytes
        elif dominant_activation is not None:
            dominant_component_category = dominant_activation.category.upper()
            dominant_component_kind = "PEAK_GROWTH"
            dominant_component_bytes = dominant_activation.peak_growth_bytes
        else:
            dominant_component_category = "UNKNOWN"
            dominant_component_kind = "UNKNOWN"
            dominant_component_bytes = 0
        logger.warning(
            "[ACTIVATION_PEAK][VERDICT] rank=%s dominant_top_level=%s "
            "top_level_growth=%s dominant_component=%s component_kind=%s "
            "component_bytes=%.3f MiB profile_absolute_peak=%.3f GiB",
            self.rank,
            dominant_top.category.upper() if dominant_top else "UNKNOWN",
            f"{dominant_top.peak_growth_bytes / _MIB:.3f} MiB" if dominant_top else "unknown",
            dominant_component_category,
            dominant_component_kind,
            dominant_component_bytes / _MIB,
            self.overall_peak_bytes / _GIB,
        )

        # Locate the single absolute memory peak across the whole profile: the
        # most-specific scope whose in-scope high-water mark reaches
        # overall_peak_bytes. Falls back to the absolute-peak winner when the
        # pinning record was never set (e.g. peak landed between checkpoints).
        global_peak_record = _most_specific_peak_record(self)
        if global_peak_record is not None:
            logger.warning(
                "[ACTIVATION_PEAK][GLOBAL_PEAK] rank=%s absolute_peak=%.3f GiB "
                "category=%s label=%s layer=%s tokens=%s growth=%.3f MiB "
                "baseline=%.3f GiB",
                self.rank,
                self.overall_peak_bytes / _GIB,
                global_peak_record.category.upper(),
                global_peak_record.label,
                "-" if global_peak_record.layer_idx is None else global_peak_record.layer_idx,
                "-" if global_peak_record.num_tokens is None else global_peak_record.num_tokens,
                global_peak_record.peak_growth_bytes / _MIB,
                global_peak_record.baseline_bytes / _GIB,
            )
        else:
            logger.warning(
                "[ACTIVATION_PEAK][GLOBAL_PEAK] rank=%s absolute_peak=%.3f GiB location=unknown",
                self.rank,
                self.overall_peak_bytes / _GIB,
            )


_ACTIVE_PROFILER: ContextVar[ActivationPeakProfiler | None] = ContextVar(
    "ascend_activation_peak_profiler",
    default=None,
)


def _most_specific_peak_record(
    profiler: ActivationPeakProfiler,
) -> ActivationPeakRecord | None:
    """Most-specific scope whose in-scope peak equals the absolute peak.

    Records are appended on scope exit in close order: inner scopes close
    before their parents, so the first record whose ``peak_bytes`` equals
    ``overall_peak_bytes`` is the innermost scope that pinned the peak. Falls
    back to ``overall_peak_record`` and then to the raw max-peak record when
    no record matches exactly (e.g. the peak landed between checkpoints).
    """
    if not profiler.records:
        return profiler.overall_peak_record
    for record in profiler.records:
        if record.peak_bytes == profiler.overall_peak_bytes:
            return record
    return profiler.overall_peak_record or max(
        profiler.records, key=lambda item: item.peak_bytes
    )


def is_activation_peak_profiling() -> bool:
    return _ACTIVE_PROFILER.get() is not None


@contextmanager
def activation_peak_profile_session(
    *,
    enabled: bool,
    rank: int,
) -> Iterator[ActivationPeakProfiler | None]:
    if not enabled:
        yield None
        return

    profiler = ActivationPeakProfiler(rank=rank)
    profiler._synchronize()
    profiler.overall_peak_bytes = int(torch.npu.max_memory_allocated())
    torch.npu.reset_peak_memory_stats()
    token = _ACTIVE_PROFILER.set(profiler)
    logger.warning(
        "[ACTIVATION_PEAK] rank=%s profiling enabled; NPU synchronization is active and intended only for debugging.",
        rank,
    )
    try:
        yield profiler
    finally:
        profiler._checkpoint()
        _ACTIVE_PROFILER.reset(token)
        profiler.log_summary()


@contextmanager
def record_activation_peak(
    category: str,
    label: str,
    *,
    layer_idx: int | None = None,
    num_tokens: int | None = None,
) -> Iterator[None]:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        yield
        return
    with profiler.record(
        category,
        label,
        layer_idx=layer_idx,
        num_tokens=num_tokens,
    ):
        yield


def record_activation_buffer(category: str, label: str, size_bytes: int) -> None:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is not None:
        profiler.record_buffer(category, label, size_bytes)


def log_ffn_chunk_decision(
    *,
    comm_method: str,
    path: str,
    enabled: bool,
    raw_tokens: int,
    dispatched_tokens: int | None,
    chunk_size: int,
    num_chunks: int,
    applied: bool,
    reason: str,
) -> None:
    """Log one MoE chunking decision inside an activation profile scope."""
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        return

    scope = next(
        (scope for scope in reversed(profiler._scopes) if scope.category in {"target_moe", "mtp_moe"}),
        profiler._scopes[-1] if profiler._scopes else None,
    )
    logger.warning(
        "[ACTIVATION_PEAK][FFN_CHUNK] rank=%s category=%s label=%s "
        "layer=%s comm=%s path=%s enabled=%s raw_tokens=%s "
        "dispatched_tokens=%s chunk_size=%s chunks=%s applied=%s reason=%s",
        profiler.rank,
        scope.category.upper() if scope is not None else "UNKNOWN",
        scope.label if scope is not None else "-",
        "-" if scope is None or scope.layer_idx is None else scope.layer_idx,
        comm_method,
        path,
        enabled,
        raw_tokens,
        "-" if dispatched_tokens is None else dispatched_tokens,
        chunk_size,
        num_chunks,
        applied,
        reason,
    )


def _current_mtp_moe_scope(profiler: ActivationPeakProfiler) -> _ActiveScope | None:
    for scope in reversed(profiler._scopes):
        if scope.category.startswith("mtp_") and ("moe" in scope.category or "ffn" in scope.category):
            return scope
    return None


@contextmanager
def record_mtp_moe_substage(
    component: str,
    label: str,
    *,
    num_tokens: int | None = None,
) -> Iterator[None]:
    """Record a detailed MTP MoE substage when inside an MTP MoE scope."""
    profiler = _ACTIVE_PROFILER.get()
    scope = _current_mtp_moe_scope(profiler) if profiler is not None else None
    if profiler is None or scope is None:
        yield
        return

    with profiler.record(
        f"mtp_moe_{component}",
        f"mtp.layer.{scope.layer_idx}.{label}",
        layer_idx=scope.layer_idx,
        num_tokens=num_tokens,
    ):
        yield


def log_mtp_moe_chunk_detail(
    *,
    chunk_idx: int,
    num_chunks: int,
    raw_start: int,
    raw_end: int,
    dispatched_tokens: int | None,
) -> None:
    """Log raw and dispatched token counts for one MTP MoE chunk."""
    profiler = _ACTIVE_PROFILER.get()
    scope = _current_mtp_moe_scope(profiler) if profiler is not None else None
    if profiler is None or scope is None:
        return

    logger.warning(
        "[ACTIVATION_PEAK][FFN_CHUNK_DETAIL] rank=%s category=MTP_MOE "
        "layer=%s chunk=%s/%s raw_range=%s:%s raw_tokens=%s dispatched_tokens=%s",
        profiler.rank,
        "-" if scope.layer_idx is None else scope.layer_idx,
        chunk_idx + 1,
        num_chunks,
        raw_start,
        raw_end,
        raw_end - raw_start,
        "-" if dispatched_tokens is None else dispatched_tokens,
    )


# ---------------------------------------------------------------------------
# Serving-path memory logging (real request serving, not profile_run)
# ---------------------------------------------------------------------------

_SERVING_MEMORY_STEPS: int = 0


def serving_memory_log_step(
    *,
    rank: int,
    phase: str,
    num_tokens: int | None,
) -> None:
    """Log one allocator snapshot during real request serving.

    Called by the model runner around the forward boundary of each
    ``execute_model`` step (e.g. ``phase="pre_forward"`` / ``"post_forward"``).
    Prints ``[SERVING_MEMORY][STEP]`` with:

    - ``allocated``: ``memory_allocated()`` — bytes currently held by the torch
      allocator (bookkeeping read, no device sync).
    - ``run_peak``: ``max_memory_allocated()`` — cumulative high-water mark
      since process start (never reset on the serving path).
    - ``step_delta``: change in ``allocated`` versus the previous phase of the
      same step (positive ⇒ this phase grew the working set).

    This function MUST NOT call ``torch.npu.synchronize()`` or
    ``reset_peak_memory_stats()``: the serving path may run under ACL graph
    capture/replay, and either call would desync the graph. Values are allowed
    to lag in-flight allocations by one step.

    ``phase="step_summary"`` is emitted once per step (after post_forward) and
    folds in the step's token count and the per-step allocated delta.
    """
    global _SERVING_MEMORY_STEPS
    allocated = int(torch.npu.memory_allocated())
    run_peak = int(torch.npu.max_memory_allocated())
    logger.warning(
        "[SERVING_MEMORY][STEP] rank=%s phase=%s step=%s tokens=%s "
        "allocated=%.3f GiB run_peak=%.3f GiB",
        rank,
        phase,
        _SERVING_MEMORY_STEPS,
        "-" if num_tokens is None else num_tokens,
        allocated / _GIB,
        run_peak / _GIB,
    )
    if phase == "post_forward":
        _SERVING_MEMORY_STEPS += 1


def serving_memory_log_summary(*, rank: int) -> None:
    """Log the final cumulative allocator peak once at worker teardown.

    Optional convenience entry point; callers (e.g. worker shutdown) may invoke
    it to surface the run-wide high-water mark without waiting for a profile.
    """
    run_peak = int(torch.npu.max_memory_allocated())
    allocated = int(torch.npu.memory_allocated())
    logger.warning(
        "[SERVING_MEMORY][SUMMARY] rank=%s steps=%s allocated=%.3f GiB run_peak=%.3f GiB",
        rank,
        _SERVING_MEMORY_STEPS,
        allocated / _GIB,
        run_peak / _GIB,
    )


# ---------------------------------------------------------------------------
# Per-step scoped serving-peak mode (real request serving, enforce_eager only)
# ---------------------------------------------------------------------------

# Display order for the per-component breakdown in a STEP_PEAK block. Attention
# sub-components first (innermost scopes), then the other top-level categories
# that may pin the peak. Categories absent from a step are skipped.
_SERVING_PEAK_COMPONENT_ORDER: tuple[str, ...] = (
    "target_attn_qkv_proj",
    "target_attn_indexer",
    "target_attn_compressor",
    "target_attn_sfa",
    "target_attn_o_proj",
    "target_attention",
    "target_moe",
    "target_sp_gather",
    "target_tail",
    "target_head",
    "target_embedding",
    "target_hc_expand",
    "target_layer",
    "target_model",
)


def _component_display_name(category: str) -> str:
    # "target_attn_qkv_proj" -> "ATTN_QKV_PROJ"; drop the "target_" prefix.
    prefix = "target_"
    name = category[len(prefix):] if category.startswith(prefix) else category
    return name.upper()


@contextmanager
def serving_peak_step_session(
    *,
    rank: int,
    num_tokens: int,
) -> Iterator[ActivationPeakProfiler]:
    """Wrap one serving forward in a throwaway activation-peak profiler.

    Sets the ``_ACTIVE_PROFILER`` ContextVar so the existing
    ``record_activation_peak`` scopes in the model/attention/MoE paths fire
    for this single forward. On exit, synchronizes, reads the step's absolute
    peak, resets the ContextVar, and logs a ``[SERVING_MEMORY][STEP_PEAK]``
    block with the absolute peak, its pinning scope, and each attention
    sub-component's absolute peak + growth delta.

    Debug-only: two synchronizes per scope per step, and resets the allocator
    peak tracker per step (so the cumulative ``max_memory_allocated`` view is
    step-local while this runs). Requires ``enforce_eager``.
    """
    profiler = ActivationPeakProfiler(rank=rank)
    profiler._synchronize()
    profiler.overall_peak_bytes = int(torch.npu.max_memory_allocated())
    torch.npu.reset_peak_memory_stats()
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        profiler._checkpoint()
        _ACTIVE_PROFILER.reset(token)
        _log_serving_peak_step(profiler, rank=rank, num_tokens=num_tokens)
        # Bump the per-step counter (the scoped path skips
        # serving_memory_log_step whose post_forward branch would otherwise
        # increment it) so STEP_PEAK ``step=`` numbering stays consistent.
        global _SERVING_MEMORY_STEPS
        _SERVING_MEMORY_STEPS += 1


def _log_serving_peak_step(
    profiler: ActivationPeakProfiler,
    *,
    rank: int,
    num_tokens: int,
) -> None:
    """Emit the per-step ``[SERVING_MEMORY][STEP_PEAK]`` summary block."""
    absolute_peak = profiler.overall_peak_bytes
    pinning = profiler.most_specific_peak_record()

    # Per-category absolute high-water mark (max peak_bytes across instances).
    category_abs: dict[str, ActivationPeakRecord] = {}
    for record in profiler.records:
        prev = category_abs.get(record.category)
        if prev is None or record.peak_bytes > prev.peak_bytes:
            category_abs[record.category] = record

    if pinning is not None:
        logger.warning(
            "[SERVING_MEMORY][STEP_PEAK] rank=%s step=%s tokens=%s "
            "absolute_peak=%.3f GiB at=%s label=%s layer=%s "
            "growth=%.3f MiB baseline=%.3f GiB",
            rank,
            _SERVING_MEMORY_STEPS,
            "-" if num_tokens is None else num_tokens,
            absolute_peak / _GIB,
            _component_display_name(pinning.category),
            pinning.label,
            "-" if pinning.layer_idx is None else pinning.layer_idx,
            pinning.peak_growth_bytes / _MIB,
            pinning.baseline_bytes / _GIB,
        )
    else:
        logger.warning(
            "[SERVING_MEMORY][STEP_PEAK] rank=%s step=%s tokens=%s "
            "absolute_peak=%.3f GiB at=unknown",
            rank,
            _SERVING_MEMORY_STEPS,
            "-" if num_tokens is None else num_tokens,
            absolute_peak / _GIB,
        )

    # Per-component breakdown (attention sub-components + the other scopes that
    # may pin the peak). One line each, in a fixed display order; absent
    # categories are skipped. ``growth`` is the in-scope transient growth at
    # that component's own peak — the memory delta attributable to it.
    for category in _SERVING_PEAK_COMPONENT_ORDER:
        record = category_abs.get(category)
        if record is None:
            continue
        logger.warning(
            "[SERVING_MEMORY][STEP_PEAK_COMP] rank=%s step=%s component=%s "
            "peak=%.3f GiB growth=%.3f MiB baseline=%.3f GiB label=%s layer=%s",
            rank,
            _SERVING_MEMORY_STEPS,
            _component_display_name(category),
            record.peak_bytes / _GIB,
            record.peak_growth_bytes / _MIB,
            record.baseline_bytes / _GIB,
            record.label,
            "-" if record.layer_idx is None else record.layer_idx,
        )
