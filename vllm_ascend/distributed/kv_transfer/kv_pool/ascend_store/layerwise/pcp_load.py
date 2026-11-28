# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore replicated PCP KV using one dynamic all-gather with in-band status."""

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
    coalesce_copy_descriptors,
    shard_window,
    slice_transfer_bounds,
)
from vllm_ascend.ops.triton.pcp_restore import checked_pcp_scatter
from vllm_ascend.utils import create_hccl_pg_options

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

# A3 and A5 share the same GVA registration and byte-copy contract.
_BUFFER_ALIGNMENT = 2 * 1024 * 1024
_GROUP_TIMEOUT = timedelta(minutes=2)
_PLAN_CACHE_ENTRIES = 2
_STATUS_BYTES = 512
_WIRE_VERSION = 2
_MIN_COPY_RUN_BYTES = 64 * 1024
_COPY_RUN_TILES = 4


class PcpLayerwiseLoad:
    """One receive thread owns reusable staging and a private PCP communicator."""

    def __init__(self, store: Backend, pcp_group: GroupCoordinator, *, prefetch: bool = False, overlap: bool = False):
        self.store = store
        self.world_size = pcp_group.world_size
        self.rank = pcp_group.rank_in_group
        self.device = torch.device("npu", store.device_id)
        profile = get_current_hardware_profile()
        self.copy_block_bytes = profile.pcp_kv_scatter_block_bytes
        self._plan_cache: OrderedDict = OrderedDict()
        self.overlap_enabled = overlap
        self.prefetch_enabled = prefetch or overlap
        self._prepared = None
        self._descriptor_capacity = 0
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
        self._gather_ready = torch.npu.Event()
        self._batch_comm_done = torch.npu.Event()
        self._capacity = 0
        self._storage = self.local = self.gathered = None
        self._result = torch.empty((1,), dtype=torch.uint8, device=self.device)
        self._host_result = torch.empty((1,), dtype=torch.uint8, device="cpu", pin_memory=self.device.type != "cpu")
        self._closed = False
        self.h2d_bytes = 0
        self.restored_bytes = 0
        logger.info(
            "PCP layerwise restore: dynamic single gather, in-band status, scatter_block=%d overlap=%s",
            self.copy_block_bytes,
            overlap,
        )

    def _ensure_capacity(self, wire_bytes: int) -> None:
        """Grow at a drained boundary; all ranks agree before using new storage.

        Every rank follows the same monotonically growing capacity sequence.
        CPU control consensus is only needed on growth, never once per restore.
        This is also safe during prefetch: no device collective is entered.
        """
        if wire_bytes <= self._capacity:
            return
        self._finish_batch()
        capacity = ((wire_bytes + _BUFFER_ALIGNMENT - 1) // _BUFFER_ALIGNMENT) * _BUFFER_ALIGNMENT
        storage = local = gathered = None
        attempted_registration = False
        error = None
        try:
            storage = torch.empty(capacity + _BUFFER_ALIGNMENT, dtype=torch.uint8, device=self.device)
            offset = (-storage.data_ptr()) % _BUFFER_ALIGNMENT
            local = storage[offset : offset + capacity]
            gathered = torch.empty(capacity * self.world_size, dtype=torch.uint8, device=self.device)
            attempted_registration = True
            self.store.register_buffer([local.data_ptr()], [capacity])
        except Exception as exc:
            error = exc
        reports = [None] * self.world_size
        try:
            dist.all_gather_object(reports, (capacity, error is None), group=self.control_group)
            if any(report != (capacity, True) for report in reports):
                raise RuntimeError(
                    "PCP staging allocation/registration failed or capacities differ across ranks"
                ) from error
        except Exception:
            if attempted_registration:
                self.store.unregister_buffer([local.data_ptr()], [capacity])
            raise
        if self.local is not None:
            self.store.unregister_buffer([self.local.data_ptr()], [self._capacity])
        self._storage, self.local, self.gathered = storage, local, gathered
        self._capacity = capacity

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
            _WIRE_VERSION,
            self._capacity,
            self.prefetch_enabled,
            self.overlap_enabled,
        )
        dist.all_gather_object(signatures, plan_identity, group=self.control_group)
        if any(value != signatures[0] for value in signatures):
            raise RuntimeError("PCP layerwise load plans differ across ranks; refusing to mix KV data")

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
                self._scatter_layout(starts, ends, end, shard_bytes),
            )
            for start, end, shard_bytes, local_start, local_end in shard_window(
                int(sizes.sum()), self.world_size, self.rank
            )
        )
        self._plan_cache[key] = windows
        if len(self._plan_cache) > _PLAN_CACHE_ENTRIES:
            self._plan_cache.popitem(last=False)
        return windows

    def _scatter_layout(self, starts: np.ndarray, ends: np.ndarray, total: int, shard_bytes: int) -> tuple:
        # Cache relative indices/offsets only. Virtual addresses are attached
        # later, so alternating layers and allocator reuse cannot stale them.
        parts = []
        for rank in range(self.world_size):
            begin, end = min(rank * shard_bytes, total), min((rank + 1) * shard_bytes, total)
            indices, offsets, packed, lengths = slice_transfer_bounds(starts, ends, begin, end)
            parts.append((indices, offsets, packed + rank * (shard_bytes + _STATUS_BYTES), lengths))
        return tuple(np.concatenate([part[i] for part in parts]) for i in range(4))

    def reset_prefetch(self) -> None:
        if self._prepared is not None:
            self.stream.synchronize()
            self._prepared = None

    def defer_prefetch_error(self, token: object, error: Exception) -> None:
        self.reset_prefetch()
        self._prepared = (token, None, None, True, error)

    def _read_plan(self, window: tuple, gvas: np.ndarray, copy_from_pool: Callable) -> tuple[bool, Exception | None]:
        _, _, shard_bytes, valid, local_slice, _ = window
        indices, offsets, packed, lengths = local_slice
        failed, failure = False, None
        try:
            result = copy_from_pool(gvas[indices] + offsets, self.local.data_ptr() + packed, lengths)
            failed = result != 0
            if not failed:
                self.h2d_bytes += int(lengths.sum())
        except Exception as exc:
            failed, failure = True, exc
        # Even a failed reader enters the ONE data collective. Never send stale
        # staging bytes, and never publish them: every scatter checks all flags.
        if failed:
            self.local[:shard_bytes].zero_()
        elif valid < shard_bytes:
            self.local[valid:shard_bytes].zero_()
        self.local[shard_bytes : shard_bytes + _STATUS_BYTES].zero_()
        if failed:
            self.local[shard_bytes : shard_bytes + 1].fill_(1)
        return failed, failure

    def prefetch(self, gvas: np.ndarray, sizes: np.ndarray, copy_from_pool: Callable, token: object) -> None:
        """Read this rank's complete shard ahead; no device collective here."""
        if not self.prefetch_enabled:
            return
        self.reset_prefetch()
        saved_gvas, saved_sizes = gvas.copy(), sizes.copy()
        try:
            plan = self._get_transfer_plan(saved_sizes)
            if not plan:
                return
            self._ensure_capacity(plan[0][2] + _STATUS_BYTES)
            with torch.npu.stream(self.stream):
                failed, failure = self._read_plan(plan[0], saved_gvas, copy_from_pool)
            self._prepared = (token, saved_gvas, saved_sizes, failed, failure)
        except Exception as exc:
            self.defer_prefetch_error(token, exc)

    def _ensure_descriptors(self, count: int) -> None:
        if count <= self._descriptor_capacity:
            return
        # Previous batches have finished before this buffer can be resized.
        # Descriptors remain live through the final stream synchronization.
        self._descriptor_capacity = max(count, 2 * self._descriptor_capacity)
        shape = (1, 3 * self._descriptor_capacity)
        self._host_descriptors = torch.empty(
            shape, dtype=torch.int64, device="cpu", pin_memory=self.device.type != "cpu"
        )
        self._device_descriptors = torch.empty(shape, dtype=torch.int64, device=self.device)

    def _upload_descriptors(self, descriptors: tuple) -> int:
        sources, destinations, lengths = descriptors
        count = len(lengths)
        if count:
            host = self._host_descriptors.view(-1)[: 3 * count].view(3, count)
            values = host.numpy()
            values[0] = self.gathered.data_ptr() + sources
            values[1] = destinations
            values[2] = lengths
            self._device_descriptors.view(-1)[: 3 * count].copy_(host.view(-1), non_blocking=True)
        return count

    def _scatter(self, count: int, shard_bytes: int) -> None:
        pointers = self._device_descriptors.view(-1)[: 3 * count].view(3, count)
        with torch.profiler.record_function("kv_pool_pcp_scatter"):
            checked_pcp_scatter(
                pointers,
                self.gathered,
                self._result,
                shard_bytes,
                self.world_size,
                _STATUS_BYTES,
                self.copy_block_bytes,
            )
        # Read status as part of the existing batch completion, never a new
        # host wait between AllGather and scatter. The kernel forbids all writes
        # on any peer's failure, including ranks with no valid local payload.
        self._host_result.copy_(self._result, non_blocking=True)

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
        prepared, self._prepared = self._prepared, None
        use_prepared = (
            prepared is not None
            and prepared[0] is token
            and (prepared[1] is None or (np.array_equal(gvas, prepared[1]) and np.array_equal(sizes, prepared[2])))
        )
        if prepared is not None and not use_prepared:
            self.stream.synchronize()
        if not plan:
            return
        assert len(plan) == 1
        window = plan[0]
        total, shard_bytes = window[1], window[2]
        self._ensure_capacity(shard_bytes + _STATUS_BYTES)
        # In-band trailers separate rank payloads. Split source descriptors at
        # every rank boundary before coalescing, so no copy traverses a trailer.
        indices, offsets, packed, lengths = window[5]
        descriptors = coalesce_copy_descriptors(
            packed,
            addrs[indices] + offsets,
            lengths,
            max(_MIN_COPY_RUN_BYTES, _COPY_RUN_TILES * self.copy_block_bytes),
        )
        self._ensure_descriptors(max(1, len(descriptors[2])))
        failure = None
        with torch.npu.stream(self.stream):
            try:
                if use_prepared:
                    failed, failure = prepared[3:]
                    if prepared[1] is None:
                        # A preparation failure may precede filling the trailer.
                        self.local[: shard_bytes + _STATUS_BYTES].zero_()
                        self.local[shard_bytes : shard_bytes + 1].fill_(1)
                else:
                    failed, failure = self._read_plan(window, gvas, copy_from_pool)
                with torch.npu.stream(self.scatter_stream):
                    count = self._upload_descriptors(descriptors)
                with torch.profiler.record_function("kv_pool_pcp_all_gather"):
                    wire_bytes = shard_bytes + _STATUS_BYTES
                    dist.all_gather_into_tensor(
                        self.gathered[: wire_bytes * self.world_size], self.local[:wire_bytes], group=self.device_group
                    )
                self._gather_ready.record(self.stream)
                self.scatter_stream.wait_event(self._gather_ready)
                with torch.npu.stream(self.scatter_stream):
                    self._scatter(count, shard_bytes)
            finally:
                self._finish_batch()
        if int(self._host_result[0]):
            raise RuntimeError("PCP layerwise pool read failed on at least one rank") from failure
        self.restored_bytes += total

    def close(self) -> None:
        if self._closed:
            return
        self._finish_batch()
        if self.local is not None:
            self.store.unregister_buffer([self.local.data_ptr()], [self._capacity])
        dist.destroy_process_group(self.device_group)
        dist.destroy_process_group(self.control_group)
        self._closed = True
