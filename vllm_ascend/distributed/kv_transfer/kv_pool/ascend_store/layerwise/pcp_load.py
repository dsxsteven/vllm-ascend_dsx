# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore replicated PCP KV using sharded pool reads and bounded all-gathers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.device.hardware_profile import get_current_hardware_profile
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pcp_load_plan import (
    SHARD_CAPACITY,
    coalesce_copy_descriptors,
    shard_window,
    slice_transfer_bounds,
)
from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.utils import create_hccl_pg_options

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

# A3 and A5 share the same GVA registration and byte-copy contract.
_BUFFER_ALIGNMENT = 2 * 1024 * 1024
_GROUP_TIMEOUT = timedelta(minutes=2)
_PLAN_CACHE_ENTRIES = 2
_PREFETCH_SHARDS = 3
# Give every shard in a batch its own receive buffer so AG2 need not wait
# for scatter0 to release storage while local LI computation is running.
_GATHER_BUFFERS = 3
_MIN_COPY_RUN_BYTES = 64 * 1024
_COPY_RUN_TILES = 4


class PcpLayerwiseLoad:
    """One receive thread owns staging and an independent PCP communicator."""

    def __init__(self, store: Backend, pcp_group: GroupCoordinator, *, prefetch: bool = False, overlap: bool = False):
        self.store = store
        self.world_size = pcp_group.world_size
        self.rank = pcp_group.rank_in_group
        self.device = torch.device("npu", store.device_id)
        profile = get_current_hardware_profile()
        self.copy_block_bytes = profile.pcp_kv_scatter_block_bytes
        self._use_device_status_reduce = profile.pcp_kv_device_status_reduce
        self._status_needs_reset = True
        self._plan_cache: OrderedDict = OrderedDict()
        self.overlap_enabled = overlap
        self.prefetch_enabled = prefetch or overlap
        self.batch_windows = _PREFETCH_SHARDS if self.prefetch_enabled else 1
        self._prepared = None
        self._control_status = torch.empty((), dtype=torch.int32, device="cpu")
        self._descriptor_capacity = 0
        self._descriptor_width = 0
        self._host_descriptors = None
        self._device_descriptors = None
        group_name = f"kv_pool_pcp_load_{pcp_group.ranks[0]}"
        # PCP groups are disjoint; all members create control then data.
        self.control_group = dist.new_group(
            ranks=pcp_group.ranks,
            backend="gloo",
            timeout=_GROUP_TIMEOUT,
            use_local_synchronization=True,
            group_desc=group_name + "_control",
        )
        options = create_hccl_pg_options(group_name)
        # Do not inherit the much larger MC2 buffer from HCCL_BUFFSIZE.
        options.hccl_config["hccl_buffer_size"] = 64
        self.device_group = dist.new_group(
            ranks=pcp_group.ranks,
            backend="hccl",
            timeout=_GROUP_TIMEOUT,
            use_local_synchronization=True,
            pg_options=options,
            group_desc=group_name,
        )
        self.stream = torch.npu.Stream(device=self.device)
        self.scatter_stream = torch.npu.Stream(device=self.device) if overlap else self.stream
        self._gather_ready = [torch.npu.Event() for _ in range(_GATHER_BUFFERS)] if overlap else []
        self._scatter_done = [torch.npu.Event() for _ in range(_GATHER_BUFFERS)] if overlap else []
        self._batch_comm_done = torch.npu.Event() if overlap else None
        self._read_status = (
            torch.empty((), dtype=torch.int32, device=self.device) if self._use_device_status_reduce else None
        )
        # Initialize outside attention. Successful MAX reductions leave zero
        # intact, so subsequent batches do not launch a competing scalar Fill.
        if self._read_status is not None:
            with torch.npu.stream(self.stream):
                self._read_status.zero_()
            self.stream.synchronize()
            self._status_needs_reset = False
        self._storage = torch.empty(
            SHARD_CAPACITY * self.batch_windows + _BUFFER_ALIGNMENT, dtype=torch.uint8, device=self.device
        )
        offset = (-self._storage.data_ptr()) % _BUFFER_ALIGNMENT
        self.locals = [
            self._storage[offset + i * SHARD_CAPACITY : offset + (i + 1) * SHARD_CAPACITY]
            for i in range(self.batch_windows)
        ]
        self.local = self.locals[0]
        self.gathered_buffers = [
            torch.empty(SHARD_CAPACITY * self.world_size, dtype=torch.uint8, device=self.device)
            for _ in range(_GATHER_BUFFERS if overlap else 1)
        ]
        self.gathered = self.gathered_buffers[0]
        self.store.register_buffer([local.data_ptr() for local in self.locals], [SHARD_CAPACITY] * len(self.locals))
        self._closed = False
        self.h2d_bytes = 0
        self.restored_bytes = 0
        logger.info(
            "PCP layerwise H2D all-gather enabled: rank=%d/%d shard_buffer=%d bytes "
            "scatter_block=%d bytes device_status_reduce=%s prefetch=%s batch_windows=%d overlap=%s gather_buffers=%d",
            self.rank,
            self.world_size,
            SHARD_CAPACITY,
            self.copy_block_bytes,
            self._use_device_status_reduce,
            self.prefetch_enabled,
            self.batch_windows,
            self.overlap_enabled,
            len(self.gathered_buffers),
        )

    def validate_plan(self, signature: str) -> None:
        """Agree once per step before enqueuing background collectives.

        Sign logical keys, block selections and sizes, not process-local GVAs.
        """
        # Called after the worker's full step drain and before lease reuse.
        # A token retains the old task list, so identity cannot alias a new step.
        self.reset_prefetch()
        signatures = [None] * self.world_size
        plan_identity = (
            signature,
            self.batch_windows,
            self._read_status is not None,
            self.prefetch_enabled,
            len(self.gathered_buffers),
            self.overlap_enabled,
        )
        dist.all_gather_object(signatures, plan_identity, group=self.control_group)
        if any(value != signatures[0] for value in signatures):
            raise RuntimeError("PCP layerwise load plans differ across ranks; refusing to mix KV data")

    def _check_copy_result(self, failed: bool) -> None:
        if self._read_status is None:
            status = self._control_status
            status.fill_(int(failed))
            group = self.control_group
        else:
            # The demand fence or bounded attention window orders this receive
            # stream against model collectives. Check every rank before scatter.
            status = self._read_status
            # An exception or a peer failure can leave a nonzero/unknown value.
            # Only a completed, successful consensus permits zero reuse.
            needs_reset = self._status_needs_reset
            self._status_needs_reset = True
            if failed or needs_reset:
                status.fill_(int(failed))
            group = self.device_group
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
        if status.item():
            raise RuntimeError("PCP layerwise pool read failed on at least one rank")
        self._status_needs_reset = False

    def _get_transfer_plan(self, sizes: np.ndarray) -> tuple:
        # Cache only relative ranges, never a GVA or device address. Snapshot
        # lengths in the key so mutating/reusing a caller array cannot reuse an
        # obsolete plan. Two entries cover alternating KV layouts without
        # retaining every prefix length seen by a long-running service.
        key = (sizes.dtype.str, sizes.shape, sizes.tobytes())
        cached = self._plan_cache.get(key)
        if cached is not None:
            self._plan_cache.move_to_end(key)
            return cached
        if np.any(sizes < 0):
            raise ValueError("Invalid packed KV byte range")
        ends = np.cumsum(sizes, dtype=np.int64)
        starts = ends - sizes
        windows = tuple(
            (
                start,
                end,
                shard_bytes,
                local_end - local_start,
                slice_transfer_bounds(starts, ends, local_start, local_end),
                slice_transfer_bounds(starts, ends, start, end),
            )
            for start, end, shard_bytes, local_start, local_end in shard_window(
                int(sizes.sum()), self.world_size, self.rank
            )
        )
        self._plan_cache[key] = windows
        if len(self._plan_cache) > _PLAN_CACHE_ENTRIES:
            self._plan_cache.popitem(last=False)
        return windows

    def reset_prefetch(self) -> None:
        if self._prepared is not None:
            self.stream.synchronize()
            self._prepared = None

    def defer_prefetch_error(self, token: object, error: Exception) -> None:
        self.reset_prefetch()
        self._prepared = (token, None, None, True, error)

    def _read_batch(self, windows: tuple, gvas: np.ndarray, copy_from_pool: Callable) -> tuple[bool, Exception | None]:
        failed = False
        failure = None
        for slot, window in enumerate(windows):
            _, _, shard_bytes, valid, local_slice, _ = window
            indices, offsets, packed, lengths = local_slice
            local = self.locals[slot]
            try:
                result = copy_from_pool(gvas[indices] + offsets, local.data_ptr() + packed, lengths)
                failed |= result != 0
                if result == 0:
                    self.h2d_bytes += int(lengths.sum())
            except Exception as exc:
                failed = True
                failure = exc
            # Padding does not overlap the synchronous backend write. Queue it
            # on the receive stream; the later gather observes it without a
            # separate CPU synchronization, including ranks with empty shards.
            if valid < shard_bytes:
                local[valid:shard_bytes].zero_()
        return failed, failure

    def prefetch(self, gvas: np.ndarray, sizes: np.ndarray, copy_from_pool: Callable, token: object) -> None:
        """Read one bounded batch locally. Never enter a process group here."""
        if not self.prefetch_enabled:
            return
        self.reset_prefetch()
        # Snapshot mutable builder views; cached metadata never owns pool data.
        saved_gvas, saved_sizes = gvas.copy(), sizes.copy()
        try:
            windows = self._get_transfer_plan(saved_sizes)[: self.batch_windows]
            with torch.npu.stream(self.stream):
                failed, failure = self._read_batch(windows, saved_gvas, copy_from_pool)
            self._prepared = (token, saved_gvas, saved_sizes, failed, failure)
        except Exception as exc:
            self.defer_prefetch_error(token, exc)

    def _ensure_descriptors(self, count: int) -> None:
        if count <= self._descriptor_capacity:
            return
        # Previous batches have finished before this buffer can be resized.
        # Each slot is used once per batch, and retained through its final sync.
        self._descriptor_capacity = max(count, 2 * self._descriptor_capacity)
        shape = (self.batch_windows, 3 * self._descriptor_capacity)
        self._host_descriptors = torch.empty(
            shape, dtype=torch.int64, device="cpu", pin_memory=self.device.type != "cpu"
        )
        self._device_descriptors = torch.empty(shape, dtype=torch.int64, device=self.device)

    def _upload_descriptors(self, descriptors: tuple) -> list[int]:
        counts = [len(lengths) for _, _, lengths in descriptors]
        width = self._descriptor_width = 3 * max(counts, default=0)
        if not width:
            return counts
        # Pack rows without allocation-capacity gaps so this remains one
        # contiguous DMA, including smaller batches after descriptor growth.
        elements = len(counts) * width
        host = self._host_descriptors.view(-1)[:elements].view(len(counts), width)
        for slot, (packed, destinations, lengths) in enumerate(descriptors):
            count = counts[slot]
            values = host[slot, : 3 * count].view(3, count).numpy()
            values[0] = self.gathered_buffers[slot % len(self.gathered_buffers)].data_ptr() + packed
            values[1] = destinations
            values[2] = lengths
        # Slot rows remain alive until the final copy-stream fence, including
        # exceptional exits. Unused row tails are never read by the kernel.
        self._device_descriptors.view(-1)[:elements].copy_(host.view(-1), non_blocking=True)
        return counts

    def _scatter(self, slot: int, count: int) -> None:
        if not count:
            return
        start = slot * self._descriptor_width
        pointers = self._device_descriptors.view(-1)[start : start + 3 * count].view(3, count)
        with torch.profiler.record_function("kv_pool_pcp_scatter"):
            batch_memcpy_kernel[(count,)](pointers[0], pointers[1], pointers[2], BLOCK_SIZE=self.copy_block_bytes)

    def _finish_batch(self) -> None:
        if not self.overlap_enabled:
            self.stream.synchronize()
            return
        # Join both streams using one host wait. Recording even after a partial
        # enqueue covers communication whose normal ready event was not reached.
        try:
            self._batch_comm_done.record(self.stream)
            self.scatter_stream.wait_event(self._batch_comm_done)
            self.scatter_stream.synchronize()
        except Exception:
            # Do not release staging/descriptor storage on an event failure.
            try:
                self.stream.synchronize()
            finally:
                self.scatter_stream.synchronize()
            raise

    def copy(
        self,
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        copy_from_pool: Callable[[np.ndarray, np.ndarray, np.ndarray], int],
        *,
        token: object = None,
    ) -> None:
        plan = self._get_transfer_plan(sizes)
        prepared = self._prepared
        self._prepared = None
        use_prepared = (
            prepared is not None
            and prepared[0] is token
            and (prepared[1] is None or (np.array_equal(gvas, prepared[1]) and np.array_equal(sizes, prepared[2])))
        )
        if prepared is not None and not use_prepared:
            self.stream.synchronize()
        max_copy_bytes = max(_MIN_COPY_RUN_BYTES, _COPY_RUN_TILES * self.copy_block_bytes)
        descriptors = tuple(
            coalesce_copy_descriptors(packed, addrs[indices] + offsets, lengths, max_copy_bytes)
            for _, _, _, _, _, (indices, offsets, packed, lengths) in plan
        )
        self._ensure_descriptors(max((len(item[2]) for item in descriptors), default=0))
        with torch.npu.stream(self.stream):
            for first in range(0, len(plan), self.batch_windows):
                windows = plan[first : first + self.batch_windows]
                try:
                    if first == 0 and use_prepared:
                        failed, failure = prepared[3:]
                    else:
                        failed, failure = self._read_batch(windows, gvas, copy_from_pool)
                    try:
                        # Consensus covers all local reads in this batch. No
                        # peer publishes any of its data if one read failed.
                        self._check_copy_result(failed)
                    except RuntimeError as exc:
                        raise exc from failure
                    with torch.npu.stream(self.scatter_stream):
                        counts = self._upload_descriptors(descriptors[first : first + len(windows)])
                    used = [False] * len(self.gathered_buffers)
                    for slot, window in enumerate(windows):
                        _, _, shard_bytes, *_ = window
                        buffer = slot % len(self.gathered_buffers)
                        if self.overlap_enabled and used[buffer]:
                            self.stream.wait_event(self._scatter_done[buffer])
                        with torch.profiler.record_function("kv_pool_pcp_all_gather"):
                            dist.all_gather_into_tensor(
                                self.gathered_buffers[buffer][: shard_bytes * self.world_size],
                                self.locals[slot][:shard_bytes],
                                group=self.device_group,
                            )
                        if self.overlap_enabled:
                            self._gather_ready[buffer].record(self.stream)
                            self.scatter_stream.wait_event(self._gather_ready[buffer])
                        with torch.npu.stream(self.scatter_stream):
                            self._scatter(slot, counts[slot])
                            if self.overlap_enabled:
                                self._scatter_done[buffer].record(self.scatter_stream)
                        used[buffer] = True
                finally:
                    # One completion fence per bounded batch. It protects all
                    # staging/descriptor reuse and the receiver's CPU event,
                    # including failure paths with work already enqueued.
                    self._finish_batch()
                self.restored_bytes += sum(window[1] - window[0] for window in windows)

    def close(self) -> None:
        if self._closed:
            return
        self._finish_batch()
        logger.info(
            "PCP layerwise load bytes: rank=%d/%d h2d=%d restored=%d",
            self.rank,
            self.world_size,
            self.h2d_bytes,
            self.restored_bytes,
        )
        self.store.unregister_buffer([local.data_ptr() for local in self.locals], [SHARD_CAPACITY] * len(self.locals))
        dist.destroy_process_group(self.device_group)
        dist.destroy_process_group(self.control_group)
        self._closed = True
