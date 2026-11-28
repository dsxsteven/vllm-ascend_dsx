# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the production kernel body with a small CPU TL semantic adapter.

This validates address/mask/status logic, not the Ascend compiler or hardware.
All pointers below refer exclusively to live CPU tensors owned by the test.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.ops.triton import pcp_restore


class Value(np.ndarray):
    def __new__(cls, data):
        return np.asarray(data).view(cls)

    def to(self, dtype):
        if isinstance(dtype, tuple):
            return Pointer(self, dtype[0])
        return Value(self.astype(dtype))


class Pointer:
    def __init__(self, addresses, dtype):
        self.addresses = np.asarray(addresses, dtype=np.int64)
        self.dtype = np.dtype(dtype)

    def __add__(self, offsets):
        return Pointer(self.addresses + offsets * self.dtype.itemsize, self.dtype)


class CpuTL:
    uint8 = np.uint8
    int32 = np.int32
    int64 = np.int64
    pid = 0

    @staticmethod
    def pointer_type(dtype):
        return (dtype,)

    def program_id(self, axis):
        return self.pid

    @staticmethod
    def arange(start, end):
        return Value(np.arange(start, end))

    @staticmethod
    def max(value, axis):
        return Value(np.max(value, axis))

    @staticmethod
    def load(pointer, mask=True, other=0, **kwargs):
        addresses = pointer.addresses
        values = np.full(addresses.shape, other, dtype=pointer.dtype)
        mask = np.broadcast_to(mask, addresses.shape)
        ctype = np.ctypeslib.as_ctypes_type(pointer.dtype)
        for index in np.ndindex(addresses.shape):
            if mask[index]:
                values[index] = ctype.from_address(int(addresses[index])).value
        return Value(values)

    @staticmethod
    def store(pointer, values, mask=True, **kwargs):
        addresses = pointer.addresses
        values, mask = np.broadcast_to(values, addresses.shape), np.broadcast_to(mask, addresses.shape)
        ctype = np.ctypeslib.as_ctypes_type(pointer.dtype)
        for index in np.ndindex(addresses.shape):
            if mask[index]:
                ctype.from_address(int(addresses[index])).value = int(values[index])


@pytest.mark.parametrize("world", [2, 3, 8, 16])
@pytest.mark.parametrize("payload", [1, 513, 4097])
@pytest.mark.parametrize("failed_rank", [None, 0, -1])
@pytest.mark.parametrize("block", [4096, 65536])
def test_kernel_body_bytes_masks_and_failure(monkeypatch, world, payload, failed_rank, block):
    trailer = 512
    wire = payload + trailer
    gathered = torch.zeros(world * wire, dtype=torch.uint8)
    expected = torch.arange(world * payload, dtype=torch.int64).to(torch.uint8).reshape(world, payload)
    packets = gathered.view(world, wire)
    packets[:, :payload].copy_(expected)
    if failed_rank is not None:
        packets[failed_rank, payload] = 7  # Any nonzero status must reject writes.
    target = torch.full((world, payload + 32), 0xA5, dtype=torch.uint8)
    pointers = torch.tensor(
        [
            [gathered.data_ptr() + rank * wire, target.data_ptr() + rank * (payload + 32) + 16, payload]
            for rank in range(world)
        ],
        dtype=torch.int64,
    ).T.contiguous()
    result = torch.full((1,), 0xFF, dtype=torch.uint8)
    adapter = CpuTL()
    monkeypatch.setattr(pcp_restore, "tl", adapter)
    kernel = pcp_restore._checked_scatter
    kernel = getattr(kernel, "fn", kernel)
    # Reverse program order also proves that writes never depend on program 0.
    for pid in reversed(range(world)):
        adapter.pid = pid
        kernel(
            *(Pointer(row.data_ptr(), np.int64) for row in pointers),
            Pointer(gathered.data_ptr(), np.uint8),
            Pointer(result.data_ptr(), np.uint8),
            world,
            payload,
            world,
            trailer,
            1 << (world - 1).bit_length(),
            block,
        )
    assert result.item() == int(failed_rank is not None)
    if failed_rank is None:
        assert torch.equal(target[:, 16:-16], expected)
        assert torch.all(target[:, :16] == 0xA5) and torch.all(target[:, -16:] == 0xA5)
    else:
        assert torch.all(target == 0xA5)


def test_empty_descriptors_still_report_status(monkeypatch):
    adapter = CpuTL()
    monkeypatch.setattr(pcp_restore, "tl", adapter)
    gathered = torch.zeros(2048, dtype=torch.uint8)
    result = torch.full((1,), 0xFF, dtype=torch.uint8)
    empty = Pointer(0, np.int64)  # Must never be dereferenced when count is zero.
    kernel = getattr(pcp_restore._checked_scatter, "fn", pcp_restore._checked_scatter)
    kernel(
        empty,
        empty,
        empty,
        Pointer(gathered.data_ptr(), np.uint8),
        Pointer(result.data_ptr(), np.uint8),
        0,
        512,
        2,
        512,
        2,
        4096,
    )
    assert result.item() == 0


def test_status_offsets_above_two_gib_do_not_wrap(monkeypatch):
    adapter = CpuTL()
    observed = []

    def load(pointer, **kwargs):
        observed.append(pointer.addresses.copy())
        return Value(np.zeros(pointer.addresses.shape, dtype=np.uint8))

    monkeypatch.setattr(adapter, "load", load)
    monkeypatch.setattr(pcp_restore, "tl", adapter)
    result = torch.full((1,), 0xFF, dtype=torch.uint8)
    # No allocation or dereference of a large buffer: only status address
    # arithmetic is intercepted. Program 0 has no copy descriptors.
    payload = 256 * 1024 * 1024
    empty = Pointer(0, np.int64)
    kernel = getattr(pcp_restore._checked_scatter, "fn", pcp_restore._checked_scatter)
    kernel(
        empty, empty, empty, Pointer(0, np.uint8), Pointer(result.data_ptr(), np.uint8), 0, payload, 16, 512, 16, 4096
    )
    assert np.array_equal(observed[0], np.arange(16, dtype=np.int64) * (payload + 512) + payload)
    assert observed[0][-1] > 2**32


def test_wrapper_rejects_invalid_layout_before_launch(monkeypatch):
    launch = SimpleNamespace(__getitem__=lambda grid: pytest.fail("Invalid input launched a kernel"))
    monkeypatch.setattr(pcp_restore, "_checked_scatter", launch)
    with pytest.raises(ValueError, match="int64 descriptors"):
        pcp_restore.checked_pcp_scatter(
            torch.empty(3, 1),
            torch.empty(2048, dtype=torch.uint8),
            torch.empty(1, dtype=torch.uint8),
            512,
            2,
            512,
            4096,
        )
