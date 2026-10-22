# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

<<<<<<< HEAD
"""Scoped activation-memory peak diagnostics for Ascend profile runs."""
=======
"""Scoped activation-memory diagnostics for eager Ascend execution.

Enable activation_peak_debug for startup profiling. serving_peak_debug and
serving_peak_debug_rank remain supported for real serving. Also enable
activation_peak_debug_serving to measure real forward AND sampling; set
activation_peak_debug_max_steps to a positive bound (0 means unlimited).
activation_peak_debug_rank selects one global rank, or -1 for every rank.

STAGE logs every measured scope. ABS_PEAK selects the largest absolute
torch-allocated peak per category, retaining THAT record's entry baseline,
growth, layer and label. SUMMARY independently selects maximum growth.
GLOBAL_PEAK identifies the leaf executing when a new session high-water mark
was observed, or UNKNOWN for uninstrumented/preexisting peaks. full_step
combines forward and sampling; RUN_PEAK survives allocator resets across steps.
Nested peaks and increments are inclusive and must not be added together.

This measures per-process torch allocator tensors, not total device HBM or all
CANN/HCCL workspaces. Synchronous eager diagnostics change overlap and timing.
Fused operators are reported as fused scopes, without invented internal peaks.
Startup preserves the historical peak for NPUWorker KV-sizing compensation;
serving uses fresh per-phase windows. All detailed collection is opt-in.
"""
>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from inspect import signature

import torch
from vllm.logger import logger

_MIB = 1024**2
_GIB = 1024**3


def _is_moe_stage(category: str) -> bool:
    """Whether a category is an MoE/FFN stage.

    MoE stages run once per layer, so their per-stage logs are far too noisy;
    they are aggregated into the summary instead of printed individually.
    """
    return "moe" in category or "ffn" in category


def _is_attention_stage(category: str) -> bool:
    """Whether a category is an attention stage (whole scope or component)."""
    return "_attn_" in category or category.endswith("_attention")


def _attn_stage_name(category: str) -> str:
    """Human readable component name, e.g. `target_attn_q_proj` -> `q_proj`."""
    return category.split("_attn_", 1)[1] if "_attn_" in category else category


def _fmt_layer(layer_idx: int | None) -> str:
    return "-" if layer_idx is None else str(layer_idx)


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
    # Category of the innermost scope active when this scope's peak was set,
    # i.e. the "driver" of the peak. None if the peak happened outside any
    # nested measured scope.
    peak_component: str | None = None


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
    # Innermost measured scope active when peak_bytes was recorded (the
    # component whose execution drove the peak); None if no nested scope was
    # active at that moment.
    peak_component: str | None = None


@dataclass
class ActivationPeakProfiler:
    """Collect nested peak measurements without losing the global profile peak."""

    rank: int
    mode: str = "profile"
    phase: str = "startup"
    step: int | None = None
    num_tokens: int | None = None
    _peak_owner: _ActiveScope | None = None
    records: list[ActivationPeakRecord] = field(default_factory=list)
    buffers: list[ActivationBufferRecord] = field(default_factory=list)
    overall_peak_bytes: int = 0
<<<<<<< HEAD
    verbose_stages: bool = False
    # Suppress per-layer/per-chunk logs; only emit the end-of-step summary.
    quiet_stages: bool = False
    # Print one line per attention layer (that layer's absolute peak, the
    # component that drove it, and the growth at that component) at the end
    # of each profiled step.
    per_layer_attn: bool = False
    _scopes: list[_ActiveScope] = field(default_factory=list)

=======
    failed: bool = False
    # The record whose in-scope peak equals overall_peak_bytes — i.e. the
    # scope instance that *pins* the absolute memory peak. Updated on scope
    # exit whenever a freshly-closed record reaches the running overall peak.
    overall_peak_record: ActivationPeakRecord | None = None
    _scopes: list[_ActiveScope] = field(default_factory=list)

    def most_specific_peak_record(self) -> ActivationPeakRecord | None:
        """Return only the actual owner of the current absolute high-water mark."""
        record = self.overall_peak_record
        return record if record is not None and record.peak_bytes == self.overall_peak_bytes else None

>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d
    def _synchronize(self) -> None:
        torch.npu.synchronize()

    def _checkpoint(self) -> int:
<<<<<<< HEAD
        self._synchronize()
        peak_bytes = int(torch.npu.max_memory_allocated())
        prev_overall = self.overall_peak_bytes
        self.overall_peak_bytes = max(self.overall_peak_bytes, peak_bytes)
        if peak_bytes > prev_overall and peak_bytes > 0:
            # This checkpoint set a new all-time peak: remember which nested
            # measured scope was executing when it happened, so the summary
            # can attribute the peak to the component that drove it.
            driver = self._scopes[-1].category if len(self._scopes) > 1 else None
            for scope in self._scopes:
                if scope.peak_bytes < peak_bytes:
                    scope.peak_bytes = peak_bytes
                    scope.peak_component = driver
        else:
            for scope in self._scopes:
                scope.peak_bytes = max(scope.peak_bytes, peak_bytes)
=======
        try:
            self._synchronize()
            peak_bytes = int(torch.npu.max_memory_allocated())
        except BaseException:
            self.failed = True
            raise
        if peak_bytes > self.overall_peak_bytes:
            self.overall_peak_bytes = peak_bytes
            # Attribute a NEW high-water mark to the currently executing leaf.
            # Parent scopes inherit its value, never its ownership. Work outside
            # an instrumented scope has an explicitly unknown location.
            self._peak_owner = self._scopes[-1] if self._scopes else None
            self.overall_peak_record = None
        for scope in self._scopes:
            scope.peak_bytes = max(scope.peak_bytes, peak_bytes)
>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d
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
        except BaseException:
            self.failed = True
            self._scopes = [active for active in self._scopes if active is not scope]
            raise
        else:
            self._checkpoint()
            allocated_after_bytes = int(torch.npu.memory_allocated())
            self._scopes.pop()
            peak_bytes = max(scope.peak_bytes, allocated_after_bytes)
            if self._scopes:
                parent = self._scopes[-1]
                if peak_bytes > parent.peak_bytes:
                    parent.peak_bytes = peak_bytes
                    parent.peak_component = scope.peak_component
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
                peak_component=scope.peak_component,
            )
            self.records.append(record)
<<<<<<< HEAD
            if self.verbose_stages and not self.quiet_stages and not _is_moe_stage(category):
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
=======
            if self._peak_owner is scope:
                self.overall_peak_record = record
            # Separate the next segment from this child's allocator high-water
            # mark. Every open ancestor already received the child's peak.
            torch.npu.reset_peak_memory_stats()
            logger.warning(
                "[ACTIVATION_PEAK][STAGE] rank=%s mode=%s phase=%s step=%s category=%s label=%s "
                "layer=%s tokens=%s baseline=%.3f GiB peak=%.3f GiB "
                "growth=%.3f MiB retained=%+.3f MiB elapsed=%.3f ms",
                self.rank,
                self.mode,
                self.phase,
                self.step,
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
>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d

    def record_buffer(self, category: str, label: str, size_bytes: int) -> None:
        if self.quiet_stages:
            return
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

    def _resolve_peak_driver(
        self,
        scope_record: ActivationPeakRecord | None,
    ) -> ActivationPeakRecord | None:
        """Return the component record that drove `scope_record`'s peak.

        `scope_record.peak_component` names the innermost measured scope active
        the instant the peak was first observed. Map it back to that component
        record (same layer, matching category). Fall back to matching by
        absolute peak when the timing-sensitive attribution is None (the peak
        first showed up on a component's entry checkpoint, before its scope was
        pushed — its later exit checkpoint observes the same peak value).

        If `scope_record` is itself a leaf component whose peak was set in its
        own scope (peak_component == its own category), it is its own driver;
        return it directly rather than searching for a (nonexistent) child.
        """
        if scope_record is None:
            return None
        if scope_record.peak_component == scope_record.category:
            return scope_record
        candidates = [
            record
            for record in self.records
            if record.layer_idx == scope_record.layer_idx and record is not scope_record
        ]
        driver = next(
            (record for record in candidates if scope_record.peak_component == record.category),
            None,
        )
        if driver is None:
            driver = next(
                (record for record in candidates if record.peak_bytes == scope_record.peak_bytes),
                None,
            )
        return driver

    def log_request_summary(self) -> None:
        """Log the focused per-request summary.

        Prints:
        - the overall NPU memory peak of the step,
        - the attention peak and the layer it occurred on,
        - the MoE peak and the layer it occurred on,
        - the per-component attention memory breakdown at the attention peak layer.
        """
        if not self.records:
            logger.warning("[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s no measured stages", self.rank)
            return

        attention_records = [
            record for record in self.records if _is_attention_stage(record.category)
        ]
        # The attention peak is the whole-attention scope (`*_attention`);
        # `*_attn_*` records are its components and must not be selected here,
        # otherwise the peak line points at a component and the component
        # breakdown lists the whole scope as one of its own parts.
        # Select by absolute peak_bytes (not growth) so the attention breakdown
        # reflects the layer where the all-time peak actually occurred, matching
        # the total_peak line. Selecting by growth would pick the layer with the
        # largest *increment* (often layer 0, whose low baseline inflates growth)
        # and misattribute the peak to a different layer than total_peak.
        whole_attention_records = [
            record for record in attention_records if record.category.endswith("_attention")
        ]
        attn_peak = max(
            whole_attention_records, key=lambda item: item.peak_bytes, default=None
        )
        overall_peak_record = max(self.records, key=lambda item: item.peak_bytes, default=None)

        # Resolve the component that drove the global absolute peak. The
        # whole-scope record's `peak_component` names the innermost measured
        # scope active the instant the all-time peak was first observed; map it
        # back to that component record to also report the growth at that
        # component. Falls back to matching by absolute peak when the timing-
        # sensitive attribution is None (peak first observed on a component's
        # entry checkpoint, before its scope was pushed).
        overall_driver = self._resolve_peak_driver(overall_peak_record)

        logger.warning(
            "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s total_peak=%.3f GiB "
            "at layer=%s label=%s component=%s growth=%.3f MiB",
            self.rank,
            self.overall_peak_bytes / _GIB,
            _fmt_layer(overall_peak_record.layer_idx) if overall_peak_record else "-",
            overall_peak_record.label if overall_peak_record else "-",
            _attn_stage_name(overall_driver.category)
            if overall_driver is not None
            else (
                _attn_stage_name(overall_peak_record.category)
                if overall_peak_record
                else "unknown"
            ),
            (overall_driver.peak_growth_bytes / _MIB)
            if overall_driver is not None
            else ((overall_peak_record.peak_growth_bytes / _MIB) if overall_peak_record else 0.0),
        )
        if attn_peak is not None:
            peak_layer_components = [
                record
                for record in attention_records
                if record.layer_idx == attn_peak.layer_idx
                and record.category != attn_peak.category
            ]
            # Determine which component the attention peak occurred in.
            # Prefer the checkpoint-time attribution (`peak_component`), which
            # records the innermost scope active the instant a new all-time peak
            # was first observed. That attribution is timing-sensitive: if the
            # new peak first shows up on a component's *entry* checkpoint (the
            # component scope has not been pushed yet, so the scope stack holds
            # only the whole-attention scope), `peak_component` is None even
            # though the peak belongs to the component that is about to run --
            # its later exit checkpoint observes the same peak value.
            #
            # Fall back to matching by absolute peak: components execute
            # sequentially, so the component whose `peak_bytes` equals the
            # attention peak is the one executing when the peak was set.
            driver = self._resolve_peak_driver(attn_peak)
            # Annotate which component (q_proj, core, o_proj, hc_post, ...) was
            # executing when the attention peak was recorded, so the peak
            # location is visible at a glance instead of only in the breakdown.
            peak_component_name = (
                _attn_stage_name(driver.category) if driver is not None else "unknown"
            )
            logger.warning(
                "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s attn_peak=%.3f GiB "
                "(growth=%.3f MiB) at layer=%s label=%s component=%s",
                self.rank,
                attn_peak.peak_bytes / _GIB,
                attn_peak.peak_growth_bytes / _MIB,
                _fmt_layer(attn_peak.layer_idx),
                attn_peak.label,
                peak_component_name,
            )
            for record in sorted(
                peak_layer_components,
                key=lambda item: item.peak_growth_bytes,
                reverse=True,
            ):
                logger.warning(
                    "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s attn component=%s "
                    "peak=%.3f GiB growth=%.3f MiB",
                    self.rank,
                    _attn_stage_name(record.category),
                    record.peak_bytes / _GIB,
                    record.peak_growth_bytes / _MIB,
                )
            if peak_layer_components:
                # The components run sequentially, so their growths do not
                # coexist and must not be summed. The attention peak
                # decomposes instead as: the growth of the component that was
                # executing when the peak was recorded, plus the baseline
                # memory already resident when that component started.
                if driver is not None:
                    carried_bytes = attn_peak.peak_bytes - driver.peak_growth_bytes
                    logger.warning(
                        "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s attn peak breakdown: "
                        "%s growth=%.3f MiB + carried=%.3f MiB (non-component residual "
                        "at component entry) = %.3f MiB total",
                        self.rank,
                        _attn_stage_name(driver.category),
                        driver.peak_growth_bytes / _MIB,
                        carried_bytes / _MIB,
                        attn_peak.peak_bytes / _MIB,
                    )
                else:
                    logger.warning(
                        "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s attn peak breakdown: "
                        "peak=%.3f MiB occurred outside measured attn components "
                        "(between components; growth values above do not sum)",
                        self.rank,
                        attn_peak.peak_bytes / _MIB,
                    )

        moe_records = [
            record
            for record in self.records
            if record.category.endswith("_moe") or "moe" in record.category or "ffn" in record.category
        ]
        # Select by absolute peak_bytes to match total_peak attribution (see
        # attn_peak selection rationale above).
        moe_peak = max(moe_records, key=lambda item: item.peak_bytes, default=None)
        if moe_peak is not None:
            logger.warning(
                "[ACTIVATION_PEAK][REQUEST_SUMMARY] rank=%s moe_peak=%.3f GiB "
                "(growth=%.3f MiB) at layer=%s label=%s",
                self.rank,
                moe_peak.peak_bytes / _GIB,
                moe_peak.peak_growth_bytes / _MIB,
                _fmt_layer(moe_peak.layer_idx),
                moe_peak.label,
            )

        # Per-layer attention peak breakdown (opt-in): for every layer, print
        # that layer's absolute attention peak, the component that drove it,
        # and the growth at that component. One line per layer so each layer's
        # peak location is visible, not just the single global attn-peak layer.
        if self.per_layer_attn:
            self._log_per_layer_attn_peaks(attention_records)

    def _log_per_layer_attn_peaks(
        self,
        attention_records: list[ActivationPeakRecord],
    ) -> None:
        """Print one line per attention layer: peak, driver component, growth."""
        # Group attention records by layer_idx. Whole-attention scopes are
        # `*_attention`; component scopes are `*_attn_*` (see
        # _is_attention_stage). Only layers with a whole-attention scope have
        # a per-layer attn peak.
        whole_by_layer: dict[int | None, ActivationPeakRecord] = {}
        components_by_layer: dict[int | None, list[ActivationPeakRecord]] = {}
        for record in attention_records:
            if record.category.endswith("_attention"):
                # Keep the highest-peak whole-attention record per layer (a
                # layer may run more than once, e.g. target vs draft reuse).
                prev = whole_by_layer.get(record.layer_idx)
                if prev is None or record.peak_bytes > prev.peak_bytes:
                    whole_by_layer[record.layer_idx] = record
            else:
                components_by_layer.setdefault(record.layer_idx, []).append(record)

        if not whole_by_layer:
            return

        logger.warning(
            "[ACTIVATION_PEAK][PER_LAYER_ATTN] rank=%s layers=%s",
            self.rank,
            len(whole_by_layer),
        )
        for layer_idx in sorted(
            whole_by_layer.keys(), key=lambda idx: (-1 if idx is None else idx)
        ):
            attn = whole_by_layer[layer_idx]
            driver = self._resolve_peak_driver(attn)
            component_name = (
                _attn_stage_name(driver.category) if driver is not None else "unknown"
            )
            # Growth at the peak location: the driver component's peak growth
            # (the increment that actually set this layer's peak). Falls back to
            # the whole-attention scope's own growth when no component is found.
            growth_mib = (
                driver.peak_growth_bytes if driver is not None else attn.peak_growth_bytes
            ) / _MIB
            logger.warning(
                "[ACTIVATION_PEAK][PER_LAYER_ATTN] rank=%s layer=%s attn_peak=%.3f GiB "
                "component=%s growth=%.3f MiB",
                self.rank,
                _fmt_layer(layer_idx),
                attn.peak_bytes / _GIB,
                component_name,
                growth_mib,
            )

    def log_summary(self) -> None:
<<<<<<< HEAD
        if not self.records and not self.buffers:
            logger.warning("[ACTIVATION_PEAK][SUMMARY] rank=%s no measured stages", self.rank)
            return

        category_winners: dict[str, ActivationPeakRecord] = {}
        for record in self.records:
            previous = category_winners.get(record.category)
            if previous is None or record.peak_growth_bytes > previous.peak_growth_bytes:
                category_winners[record.category] = record

        for category, record in sorted(category_winners.items()):
            logger.warning(
                "[ACTIVATION_PEAK][SUMMARY] rank=%s category=%s max_growth=%.3f MiB (%.3f GiB) "
                "absolute_peak=%.3f GiB label=%s layer=%s tokens=%s",
=======
        growth_winners: dict[str, ActivationPeakRecord] = {}
        absolute_winners: dict[str, ActivationPeakRecord] = {}
        for record in self.records:
            old = growth_winners.get(record.category)
            if old is None or record.peak_growth_bytes > old.peak_growth_bytes:
                growth_winners[record.category] = record
            old = absolute_winners.get(record.category)
            if (
                old is None
                or record.peak_bytes > old.peak_bytes
                or (record is self.overall_peak_record and record.peak_bytes == old.peak_bytes)
            ):
                absolute_winners[record.category] = record
        for category, record in sorted(growth_winners.items()):
            logger.warning(
                "[ACTIVATION_PEAK][SUMMARY] rank=%s mode=%s phase=%s step=%s "
                "category=%s max_growth=%.3f MiB absolute_peak=%.3f GiB "
                "label=%s layer=%s tokens=%s",
>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d
                self.rank,
                self.mode,
                self.phase,
                self.step,
                category.upper(),
                record.peak_growth_bytes / _MIB,
                record.peak_growth_bytes / _GIB,
                record.peak_bytes / _GIB,
                record.label,
                record.layer_idx,
                record.num_tokens,
            )
<<<<<<< HEAD

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
=======
        for category, record in sorted(absolute_winners.items()):
            logger.warning(
                "[ACTIVATION_PEAK][ABS_PEAK] rank=%s mode=%s phase=%s step=%s "
                "category=%s absolute_peak=%.3f GiB growth_at_peak=%.3f MiB "
                "baseline=%.3f GiB peak_bytes=%s baseline_bytes=%s growth_bytes=%s "
                "label=%s layer=%s tokens=%s",
                self.rank,
                self.mode,
                self.phase,
                self.step,
                category.upper(),
                record.peak_bytes / _GIB,
                record.peak_growth_bytes / _MIB,
                record.baseline_bytes / _GIB,
                record.peak_bytes,
                record.baseline_bytes,
                record.peak_growth_bytes,
                record.label,
                record.layer_idx,
                record.num_tokens,
            )
        record = self.overall_peak_record
        if record is not None and record.peak_bytes != self.overall_peak_bytes:
            record = None
>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d
        logger.warning(
            "[ACTIVATION_PEAK][GLOBAL_PEAK] rank=%s mode=%s phase=%s step=%s "
            "absolute_peak=%.3f GiB peak_bytes=%s location_known=%s "
            "category=%s label=%s layer=%s tokens=%s "
            "baseline_bytes=%s growth_bytes=%s",
            self.rank,
            self.mode,
            self.phase,
            self.step,
            self.overall_peak_bytes / _GIB,
            self.overall_peak_bytes,
            record is not None,
            record.category.upper() if record else "UNKNOWN",
            record.label if record else "unscoped_or_preexisting",
            record.layer_idx if record else None,
            record.num_tokens if record else self.num_tokens,
            record.baseline_bytes if record else None,
            record.peak_growth_bytes if record else None,
        )
<<<<<<< HEAD

        # Per-layer attention peak breakdown (opt-in), same as the request
        # summary. log_summary is used by the startup profile_run path.
        if self.per_layer_attn:
            attention_records = [
                record for record in self.records if _is_attention_stage(record.category)
            ]
            self._log_per_layer_attn_peaks(attention_records)
=======
        for buffer in self.buffers:
            logger.warning(
                "[ACTIVATION_PEAK][SUMMARY] rank=%s category=%s persistent=%.3f MiB label=%s",
                self.rank,
                buffer.category.upper(),
                buffer.size_bytes / _MIB,
                buffer.label,
            )
>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d


_ACTIVE_PROFILER: ContextVar[ActivationPeakProfiler | None] = ContextVar(
    "ascend_activation_peak_profiler",
    default=None,
)


def is_activation_peak_profiling() -> bool:
    return _ACTIVE_PROFILER.get() is not None


@contextmanager
def activation_peak_profile_session(
    *,
    enabled: bool,
    rank: int,
<<<<<<< HEAD
    verbose_stages: bool = True,
    per_layer_attn: bool = False,
=======
    mode: str = "profile",
    phase: str = "startup",
    step: int | None = None,
    num_tokens: int | None = None,
>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d
) -> Iterator[ActivationPeakProfiler | None]:
    if not enabled:
        yield None
        return

<<<<<<< HEAD
    profiler = ActivationPeakProfiler(
        rank=rank, verbose_stages=verbose_stages, per_layer_attn=per_layer_attn
    )
=======
    profiler = ActivationPeakProfiler(rank=rank, mode=mode, phase=phase, step=step, num_tokens=num_tokens)
>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d
    profiler._synchronize()
    # Startup preserves the historical peak for KV sizing. Serving starts a
    # fresh observation window; RUN_PEAK is maintained separately on the runner.
    profiler.overall_peak_bytes = int(
        torch.npu.max_memory_allocated() if mode == "profile" else torch.npu.memory_allocated()
    )
    torch.npu.reset_peak_memory_stats()
    token = _ACTIVE_PROFILER.set(profiler)
    logger.warning(
        "[ACTIVATION_PEAK][SESSION] rank=%s mode=%s phase=%s step=%s tokens=%s "
        "metric=torch_allocated sync=true; diagnostic timing is not production latency",
        rank,
        mode,
        phase,
        step,
        num_tokens,
    )
    try:
        yield profiler
    except BaseException:
        profiler.failed = True
        raise
    finally:
        try:
            if not profiler.failed:
                profiler._checkpoint()
        finally:
            _ACTIVE_PROFILER.reset(token)
            profiler._scopes.clear()
        if not profiler.failed:
            profiler.log_summary()


@contextmanager
def request_activation_peak_session(
    *,
    enabled: bool,
    rank: int,
    num_tokens: int,
    per_layer_attn: bool = False,
) -> Iterator[None]:
    """Per-forward activation peak session for real (served) requests.

    Unlike `activation_peak_profile_session` (used by the startup dummy
    runs), this wraps one real scheduler step and prints the per-category
    summary exactly once per invocation, without the per-stage detail log.
    """
    if not enabled:
        yield
        return

    profiler = ActivationPeakProfiler(
        rank=rank, quiet_stages=True, per_layer_attn=per_layer_attn
    )
    profiler._synchronize()
    profiler.overall_peak_bytes = int(torch.npu.max_memory_allocated())
    torch.npu.reset_peak_memory_stats()
    token = _ACTIVE_PROFILER.set(profiler)
    logger.warning(
        "[ACTIVATION_PEAK] rank=%s serving-step profiling started (num_tokens=%s); "
        "NPU synchronization is active and intended only for debugging.",
        rank,
        num_tokens,
    )
    try:
        yield
    finally:
        profiler._checkpoint()
        _ACTIVE_PROFILER.reset(token)
        logger.warning(
            "[ACTIVATION_PEAK][REQUEST] rank=%s num_tokens=%s absolute_peak=%.3f GiB",
            rank,
            num_tokens,
            profiler.overall_peak_bytes / _GIB,
        )
        profiler.log_request_summary()


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
    if profiler is None or profiler.quiet_stages:
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
    if profiler is None or profiler.quiet_stages:
        return
    scope = _current_mtp_moe_scope(profiler)
    if scope is None:
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
<<<<<<< HEAD
=======


def record_activation_substage(
    component: str,
    label: str | None = None,
    *,
    num_tokens: int | None = None,
    layer_idx: int | None = None,
):
    """Inherit target/MTP identity and layer from the actual enclosing forward."""
    profiler = _ACTIVE_PROFILER.get()
    parent = profiler._scopes[-1] if profiler and profiler._scopes else None
    phase = "mtp" if parent and parent.category.startswith("mtp_") else "target"
    inherited_layer = (
        next(
            (scope.layer_idx for scope in reversed(profiler._scopes) if scope.layer_idx is not None),
            None,
        )
        if profiler
        else None
    )
    if num_tokens is None and parent is not None:
        num_tokens = parent.num_tokens
    resolved_layer = layer_idx if layer_idx is not None else inherited_layer
    return record_activation_peak(
        f"{phase}_{component}",
        label or f"{phase}.layer.{resolved_layer}.{component}",
        layer_idx=resolved_layer,
        num_tokens=num_tokens,
    )


def log_activation_skip(component: str, reason: str) -> None:
    profiler = _ACTIVE_PROFILER.get()
    if profiler is not None:
        parent = profiler._scopes[-1] if profiler._scopes else None
        logger.warning(
            "[ACTIVATION_PEAK][SKIP] rank=%s mode=%s phase=%s step=%s component=%s reason=%s parent=%s",
            profiler.rank,
            profiler.mode,
            profiler.phase,
            profiler.step,
            component,
            reason,
            parent.label if parent else "unknown",
        )


def profile_activation(component: str):
    """Measure a method only while an explicit eager diagnostic session is active."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not is_activation_peak_profiling():
                return function(*args, **kwargs)
            tensor = next(
                (value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor) and value.ndim > 0),
                None,
            )
            with record_activation_substage(
                component,
                num_tokens=tensor.shape[0] if tensor is not None else None,
            ):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def record_moe_substage(component: str, label: str, *, num_tokens: int | None = None):
    """Measure shared target/draft MoE paths, including dispatch and each FFN chunk."""
    profiler = _ACTIVE_PROFILER.get()
    scopes = profiler._scopes if profiler is not None else []
    phase = "mtp" if scopes and scopes[-1].category.startswith("mtp_") else "target"
    layer = next((scope.layer_idx for scope in reversed(scopes) if scope.layer_idx is not None), None)
    return record_activation_substage(f"moe_{component}", f"{phase}.layer.{layer}.{label}", num_tokens=num_tokens)


def profile_runner_phase(phase: str):
    """Eager serving diagnostics owned by each runner, with no process-global counter.

    Forward and sampling are separate upstream entry points. Their records are
    merged as full_step, and a monotonic RUN_PEAK survives allocator peak resets.
    """

    def decorate(function):
        call_signature = signature(function)

        @wraps(function)
        def wrapped(self, *args, **kwargs):
            config = self.ascend_config
            enabled = getattr(config, "activation_peak_debug_serving", False) or getattr(
                config, "serving_peak_debug", False
            )
            if not enabled or is_activation_peak_profiling():
                return function(self, *args, **kwargs)
            bound = call_signature.bind_partial(self, *args, **kwargs)
            if bound.arguments.get("dummy_run", False) or bound.arguments.get("is_profile", False):
                return function(self, *args, **kwargs)
            if phase != "forward" and hasattr(self, "execute_model_state") and self.execute_model_state is None:
                return function(self, *args, **kwargs)
            scheduler = bound.arguments.get("scheduler_output")
            num_tokens = getattr(scheduler, "total_num_scheduled_tokens", None)
            if phase == "forward":
                if num_tokens == 0:
                    return function(self, *args, **kwargs)
                self._activation_peak_step = getattr(self, "_activation_peak_step", 0) + 1
                self._activation_peak_step_records = None
            step = getattr(self, "_activation_peak_step", 0)
            if not step:
                return function(self, *args, **kwargs)
            maximum = getattr(config, "activation_peak_debug_max_steps", 0)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            configured_rank = (
                config.serving_peak_debug_rank
                if getattr(config, "serving_peak_debug", False)
                else config.activation_peak_debug_rank
            )
            if (maximum and step > maximum) or (configured_rank != -1 and configured_rank != rank):
                return function(self, *args, **kwargs)
            if num_tokens is None:
                num_tokens = getattr(self, "_activation_peak_tokens", None)
            self._activation_peak_tokens = num_tokens
            with (
                activation_peak_profile_session(
                    enabled=True,
                    rank=rank,
                    mode="serving",
                    phase=phase,
                    step=step,
                    num_tokens=num_tokens,
                ) as profiler,
                record_activation_peak(
                    "target_model" if phase == "forward" else "target_sampling",
                    f"target.{phase}",
                    num_tokens=num_tokens,
                ),
            ):
                result = function(self, *args, **kwargs)
            assert profiler is not None
            previous = getattr(self, "_activation_peak_step_records", None)
            if previous is not None:
                profiler.records = previous.records + profiler.records
                profiler.buffers = previous.buffers + profiler.buffers
                if previous.overall_peak_bytes > profiler.overall_peak_bytes or (
                    previous.overall_peak_bytes == profiler.overall_peak_bytes
                    and previous.overall_peak_record is not None
                ):
                    profiler.overall_peak_bytes = previous.overall_peak_bytes
                    profiler.overall_peak_record = previous.overall_peak_record
                profiler.phase = "full_step"
                profiler.log_summary()
            self._activation_peak_step_records = profiler
            run_peak = getattr(self, "_activation_peak_run_result", None)
            if run_peak is None or profiler.overall_peak_bytes > run_peak.overall_peak_bytes:
                self._activation_peak_run_result = profiler
                record = profiler.overall_peak_record
                logger.warning(
                    "[ACTIVATION_PEAK][RUN_PEAK] rank=%s step=%s phase=%s "
                    "peak_bytes=%s category=%s label=%s layer=%s "
                    "baseline_bytes=%s growth_bytes=%s",
                    rank,
                    step,
                    phase,
                    profiler.overall_peak_bytes,
                    record.category.upper() if record else "UNKNOWN",
                    record.label if record else "unscoped_or_preexisting",
                    record.layer_idx if record else None,
                    record.baseline_bytes if record else None,
                    record.peak_growth_bytes if record else None,
                )
            return result

        return wrapped

    return decorate


def serving_memory_log_step(*, rank: int, phase: str, num_tokens: int | None) -> None:
    """Lightweight snapshot; allocator_peak is since the latest reset, not process start."""
    profiler = _ACTIVE_PROFILER.get()
    logger.warning(
        "[SERVING_MEMORY][STEP] rank=%s phase=%s step=%s tokens=%s "
        "allocated=%.3f GiB allocator_peak_since_reset=%.3f GiB",
        rank,
        phase,
        profiler.step if profiler else None,
        num_tokens,
        int(torch.npu.memory_allocated()) / _GIB,
        int(torch.npu.max_memory_allocated()) / _GIB,
    )


def serving_memory_log_summary(*, rank: int) -> None:
    """Compatibility snapshot; monotonic serving peaks are emitted by the runner."""
    serving_memory_log_step(rank=rank, phase="summary", num_tokens=None)


# Compatibility entry point retained for existing serving_peak_debug callers.
# Model runners use profile_runner_phase so sampling and the step limit are
# included, and the step state belongs to each runner rather than this counter.
_SERVING_MEMORY_STEPS = 0


def _most_specific_peak_record(profiler: ActivationPeakProfiler) -> ActivationPeakRecord | None:
    return profiler.most_specific_peak_record()


@contextmanager
def serving_peak_step_session(*, rank: int, num_tokens: int) -> Iterator[ActivationPeakProfiler]:
    """Profile one eager forward, or reuse an enclosing runner session."""
    active = _ACTIVE_PROFILER.get()
    if active is not None:
        yield active
        return
    global _SERVING_MEMORY_STEPS
    with activation_peak_profile_session(
        enabled=True,
        rank=rank,
        mode="serving",
        phase="forward",
        step=_SERVING_MEMORY_STEPS,
        num_tokens=num_tokens,
    ) as profiler:
        assert profiler is not None
        yield profiler
    _log_serving_peak_step(profiler, rank=rank, num_tokens=num_tokens)
    _SERVING_MEMORY_STEPS += 1


def _component_display_name(category: str) -> str:
    # "target_attn_qkv_proj" -> "ATTN_QKV_PROJ"; drop the "target_" prefix.
    prefix = "target_"
    name = category[len(prefix) :] if category.startswith(prefix) else category
    return name.upper()


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
            "[SERVING_MEMORY][STEP_PEAK] rank=%s step=%s tokens=%s absolute_peak=%.3f GiB at=unknown",
            rank,
            _SERVING_MEMORY_STEPS,
            "-" if num_tokens is None else num_tokens,
            absolute_peak / _GIB,
        )

    # Per-component breakdown (attention sub-components + the other scopes that
    # may pin the peak). One line each, in a fixed display order; absent
    # categories are skipped. ``growth`` is the in-scope transient growth at
    # that component's own peak — the memory delta attributable to it.
    for category in sorted(category_abs):
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
>>>>>>> 72e0c3fc79b615e39bed2ad39f2a7adfa7172d1d
