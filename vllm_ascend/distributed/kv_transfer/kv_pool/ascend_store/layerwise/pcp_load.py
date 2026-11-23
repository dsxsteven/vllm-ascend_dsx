# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore replicated PCP KV using sharded pool reads and bounded all-gathers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pcp_load_plan import (
    SHARD_CAPACITY,
    shard_window,
    slice_transfer,
)
from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.utils import create_hccl_pg_options

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

# A3 and A5 share the same GVA registration and byte-copy contract.
_BUFFER_ALIGNMENT = 2 * 1024 * 1024
_COPY_BLOCK_BYTES = 4096
_GROUP_TIMEOUT = timedelta(minutes=2)


class PcpLayerwiseLoad:
    """One receive thread owns staging and an independent PCP communicator."""

    def __init__(self, store: Backend, pcp_group: GroupCoordinator):
        self.store = store
        self.world_size = pcp_group.world_size
        self.rank = pcp_group.rank_in_group
        self.device = torch.device("npu", store.device_id)
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
        self._storage = torch.empty(SHARD_CAPACITY + _BUFFER_ALIGNMENT, dtype=torch.uint8, device=self.device)
        offset = (-self._storage.data_ptr()) % _BUFFER_ALIGNMENT
        self.local = self._storage[offset : offset + SHARD_CAPACITY]
        self.gathered = torch.empty(SHARD_CAPACITY * self.world_size, dtype=torch.uint8, device=self.device)
        self.store.register_buffer([self.local.data_ptr()], [SHARD_CAPACITY])
        self._closed = False
        self.h2d_bytes = 0
        self.restored_bytes = 0
        logger.info(
            "PCP layerwise H2D all-gather enabled: rank=%d/%d shard_buffer=%d bytes",
            self.rank,
            self.world_size,
            SHARD_CAPACITY,
        )

    def validate_plan(self, signature: str) -> None:
        """Agree once per step before enqueuing background collectives.

        Sign logical keys, block selections and sizes, not process-local GVAs.
        """
        signatures = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.control_group)
        if any(value != signatures[0] for value in signatures):
            raise RuntimeError("PCP layerwise load plans differ across ranks; refusing to mix KV data")

    def _check_copy_result(self, failed: bool) -> None:
        status = torch.tensor(int(failed), dtype=torch.int32, device="cpu")
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=self.control_group)
        if status.item():
            raise RuntimeError("PCP layerwise pool read failed on at least one rank")

    def copy(
        self,
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        copy_from_pool: Callable[[np.ndarray, np.ndarray, np.ndarray], int],
    ) -> None:
        total_bytes = int(sizes.sum())
        with torch.npu.stream(self.stream):
            for start, end, shard_bytes, local_start, local_end in shard_window(
                total_bytes, self.world_size, self.rank
            ):
                indices, offsets, packed, lengths = slice_transfer(sizes, local_start, local_end)
                # Clear collective padding, also for ranks with an empty shard.
                valid = local_end - local_start
                if valid < shard_bytes:
                    self.local[valid:shard_bytes].zero_()
                    self.stream.synchronize()
                failure = None
                try:
                    result = copy_from_pool(gvas[indices] + offsets, self.local.data_ptr() + packed, lengths)
                except Exception as exc:
                    failure = exc
                    result = -1
                # A local read failure must not strand peers in all-gather or
                # publish an apparently complete cache containing stale bytes.
                try:
                    self._check_copy_result(result != 0)
                except RuntimeError as exc:
                    raise exc from failure
                self.h2d_bytes += int(lengths.sum())
                with torch.profiler.record_function("kv_pool_pcp_all_gather"):
                    dist.all_gather_into_tensor(
                        self.gathered[: shard_bytes * self.world_size],
                        self.local[:shard_bytes],
                        group=self.device_group,
                    )
                indices, offsets, packed, lengths = slice_transfer(sizes, start, end)
                descriptors = np.stack((self.gathered.data_ptr() + packed, addrs[indices] + offsets, lengths))
                pointers = torch.from_numpy(descriptors).to(self.device)
                with torch.profiler.record_function("kv_pool_pcp_scatter"):
                    batch_memcpy_kernel[(len(indices),)](
                        pointers[0], pointers[1], pointers[2], BLOCK_SIZE=_COPY_BLOCK_BYTES
                    )
                # The receive thread publishes a CPU event. Finish scatter and
                # HCCL before attention or a subsequent buffer reuse sees it.
                self.stream.synchronize()
                self.restored_bytes += end - start

    def close(self) -> None:
        if self._closed:
            return
        self.stream.synchronize()
        logger.info(
            "PCP layerwise load bytes: rank=%d/%d h2d=%d restored=%d",
            self.rank,
            self.world_size,
            self.h2d_bytes,
            self.restored_bytes,
        )
        self.store.unregister_buffer([self.local.data_ptr()], [SHARD_CAPACITY])
        dist.destroy_process_group(self.device_group)
        dist.destroy_process_group(self.control_group)
        self._closed = True
