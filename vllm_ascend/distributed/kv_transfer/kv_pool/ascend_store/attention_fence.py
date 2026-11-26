#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager

import torch

_lock = threading.RLock()
_attention_compute_start_gate: AttentionComputeStartGate | None = None


class AttentionComputeStartGate:
    """Gate that opens when the compute stream reaches attention.

    The attention worker records an NPU event immediately before submitting the
    attention op. MemCache worker threads wait for that event to complete before
    submitting H2D/L2G work, so transfer starts when the compute stream is
    actually at the attention boundary rather than merely after the Python call
    site was reached.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.on_start: Callable[[], None] | None = None
        self.on_finish: Callable[[], None] | None = None
        # Only explicitly bounded, collective-free attention implementations
        # may open this window. A bare record() must never launch PCP collectives.
        self.on_overlap_start: Callable[[], None] | None = None
        self.on_overlap_finish: Callable[[], None] | None = None
        self._event: torch.npu.Event | None = None
        self.indexer_scope_depth = 0
        self._overlap_started = False
        self._overlap_finished = False

    def start_overlap(self) -> None:
        """Start once per layer, even when LI and SFA share the window."""
        if not self._overlap_started and self.on_overlap_start is not None:
            # Mark before invoking: an enqueue failure may leave work to join.
            self._overlap_started = True
            self.on_overlap_start()

    def finish_overlap(self) -> None:
        if self._overlap_started and not self._overlap_finished:
            self._overlap_finished = True
            if self.on_overlap_finish is not None:
                self.on_overlap_finish()

    def record(
        self,
        stream: torch.npu.Stream | None = None,
    ) -> None:
        stream = stream or torch.npu.current_stream()
        event = torch.npu.Event()
        event.record(stream)
        with self._condition:
            if self._event is None:
                self._event = event
                self._condition.notify_all()
                if self.on_start is not None:
                    self.on_start()

    def cancel(self) -> None:
        """Release gated tasks after their caller has set its transfer abort flag."""
        self.on_start = None
        self.on_finish = None
        self.on_overlap_start = None
        self.on_overlap_finish = None
        self.record()

    def wait(self, timeout: float = 10.0) -> bool:
        with self._condition:
            while self._event is None:
                if not self._condition.wait(timeout=timeout):
                    return False
            event = self._event

        event.synchronize()
        return True


def reset_attention_compute_start_gate() -> AttentionComputeStartGate:
    """Create a new per-layer gate for MemCache work.

    Layerwise prefetch tasks keep a reference to the gate that was current when
    they were submitted. The attention path opens that same gate when attention
    compute is about to be launched.
    """
    global _attention_compute_start_gate
    gate = AttentionComputeStartGate()
    with _lock:
        _attention_compute_start_gate = gate
    return gate


def get_attention_compute_start_gate() -> AttentionComputeStartGate:
    with _lock:
        gate = _attention_compute_start_gate
    if gate is None:
        gate = reset_attention_compute_start_gate()
    return gate


def record_attention_compute_start() -> None:
    """Record the compute-stream boundary immediately before attention."""
    with _lock:
        gate = _attention_compute_start_gate
    if gate is not None:
        gate.record()


@contextmanager
def indexer_attention_transfer_window():
    """Bound an optional local-LI start through the following SFA submission.

    Entering this scope does not start transfers: the indexer must first finish
    its cache writes and model collectives. Keep the legacy save gate at SFA.
    The outer finally also joins work if LI raises before reaching SFA.
    """
    with _lock:
        gate = _attention_compute_start_gate
    if gate is None:
        yield
        return
    gate.indexer_scope_depth += 1
    try:
        yield
    finally:
        gate.indexer_scope_depth -= 1
        if gate.indexer_scope_depth == 0:
            gate.finish_overlap()


def start_local_indexer_transfer_overlap() -> None:
    """Open PCP restore only inside an explicitly bounded LI/SFA scope."""
    with _lock:
        gate = _attention_compute_start_gate
    if gate is not None and gate.indexer_scope_depth:
        gate.start_overlap()


@contextmanager
def attention_transfer_window(*, allow_pcp_overlap: bool = False):
    """Run the configured transfer completion policy after attention submission.

    The event is recorded after cache writes and any preceding collectives. The
    host then submits the attention kernel. The registered connector callback
    decides whether to drain, bound, or retain its asynchronous transfer work.
    """
    with _lock:
        gate = _attention_compute_start_gate
    if gate is None:
        yield
        return
    try:
        gate.record()
        if allow_pcp_overlap:
            gate.start_overlap()
        yield
    finally:
        try:
            if allow_pcp_overlap:
                gate.finish_overlap()
        finally:
            if gate.on_finish is not None:
                gate.on_finish()
