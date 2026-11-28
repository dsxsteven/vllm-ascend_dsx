# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU validation prepared for later execution, never part of CPU-only runs."""

import pytest
import torch

from vllm_ascend.device.hardware_profile import get_current_hardware_profile
from vllm_ascend.ops.triton.pcp_restore import checked_pcp_scatter


@pytest.mark.parametrize("world", [2, 3, 8, 16])
@pytest.mark.parametrize("payload", [512, 4096, 65536])
@pytest.mark.parametrize("failed_rank", [None, 0, -1])
@pytest.mark.parametrize("empty", [False, True])
def test_checked_restore_bytes_and_failure(world, payload, failed_rank, empty):
    wire = payload + 512
    gathered = torch.zeros(world * wire, dtype=torch.uint8, device="npu")
    expected = torch.arange(world * payload, dtype=torch.int64).to(torch.uint8).reshape(world, payload)
    packets = gathered.view(world, wire)
    packets[:, :payload].copy_(expected.to("npu"))
    if failed_rank is not None:
        packets[failed_rank, payload] = 1
    target = torch.full((world, payload + 32), 0xA5, dtype=torch.uint8, device="npu")
    descriptors = (
        []
        if empty
        else [
            [gathered.data_ptr() + rank * wire, target.data_ptr() + rank * (payload + 32) + 16, payload]
            for rank in range(world)
        ]
    )
    pointers = torch.tensor(descriptors, dtype=torch.int64, device="npu").reshape(-1, 3).T.contiguous()
    result = torch.empty(1, dtype=torch.uint8, device="npu")
    checked_pcp_scatter(
        pointers, gathered, result, payload, world, 512, get_current_hardware_profile().pcp_kv_scatter_block_bytes
    )
    assert result.cpu().item() == int(failed_rank is not None)
    actual = target.cpu()
    if empty or failed_rank is not None:
        assert torch.all(actual == 0xA5)
    else:
        assert torch.equal(actual[:, 16:-16], expected)
        assert torch.all(actual[:, :16] == 0xA5) and torch.all(actual[:, -16:] == 0xA5)
