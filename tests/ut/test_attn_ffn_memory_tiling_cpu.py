# SPDX-License-Identifier: Apache-2.0
"""CPU equivalence of the candidate operator code, with TP/EP collectives emulated."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend"


def methods(path, cls, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    keep = [item for item in base.body if isinstance(item, ast.FunctionDef) and item.name in names]
    assert len(keep) == len(names)
    for item in keep:
        item.decorator_list = []
    base.body, base.bases, base.decorator_list = keep, [], []
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), base]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[cls]


def projection(tp, rows, cap):
    # Integer-valued inputs keep BF16 comparisons exact across CPU GEMM shapes.
    generator = torch.Generator().manual_seed(rows)
    x = torch.randint(-2, 3, (tp * rows, 1, 4), generator=generator).bfloat16()
    weights = torch.randint(-2, 3, (tp, 4, 8), generator=generator).bfloat16()
    parts = [x[:, 0] @ w for w in weights]
    calls = []
    for rank in range(tp):
        offset = 0

        def scatter(part, rank=rank):
            nonlocal offset
            n = len(part) // tp
            peers = [p.view(tp, rows, -1)[:, offset : offset + n].reshape(tp * n, -1) for p in parts]
            assert torch.equal(part, peers[rank]) and len(part) <= cap
            offset += n
            calls.append(len(part))
            return torch.stack(peers).sum(0).reshape(tp, n, -1)[rank]

        ns = dict(
            torch=torch,
            sp_reduce_scatter=scatter,
            torch_npu=SimpleNamespace(npu_transpose_batchmatmul=lambda value, weight, **kw: value.clone()),
        )
        cls = methods(ROOT / "attention/context_parallel/dsa_cp.py", "AscendDSACPImpl", {"_forward_o_proj"}, ns)
        impl = cls()
        impl.tp_size, impl.n_local_groups, impl.support_fp8_attention = tp, 1, False
        impl._get_batched_wo_a_weight = lambda groups: None
        impl._apply_wo_b = lambda tile, full, rank=rank: tile @ weights[rank]
        out = torch.empty(rows, 8, dtype=torch.bfloat16)
        assert impl._forward_o_proj(x, out, sp_chunk_size=cap) is out
        expected = torch.stack(parts).sum(0).reshape(tp, rows, -1)[rank]
        assert offset == rows and torch.equal(out, expected)
    return max(calls)


def moe(splits, mapping):
    ranks = len(splits)
    width = 4096
    generator = torch.Generator().manual_seed(820)
    # splits[src][dst] is the number of expert rows sent back to dst.
    sources = [torch.randn(sum(s), width, generator=generator).bfloat16() for s in splits]
    perms = [torch.randperm(len(x), generator=generator) for x in sources]
    expected_sources = [x[perm] if mapping else x for x, perm in zip(sources, perms)]
    for rank in range(ranks):
        chunks = [s.split(sizes)[rank] for s, sizes in zip(expected_sources, splits)]
        expected = torch.cat(chunks)
        tile_index = 0

        def exchange(send, receive_splits, send_splits, ep_group, rank=rank, expected=expected):
            nonlocal tile_index
            left = 1024 * tile_index
            assert list(receive_splits) == [s[rank] for s in splits] and list(send_splits) == splits[rank]
            assert torch.equal(send, expected_sources[rank][:, left : left + 1024])
            out = torch.empty(len(expected), 1024, dtype=send.dtype)
            tile_index += 1
            return send, out, SimpleNamespace(wait=lambda: out.copy_(expected[:, left : left + 1024]))

        cls = methods(
            ROOT / "ops/fused_moe/token_dispatcher.py",
            "TokenDispatcherWithAll2AllV",
            {"exchange_combine_columns", "prepare_combine_input"},
            dict(
                torch=torch,
                async_all_to_all=exchange,
                torch_npu=SimpleNamespace(npu_moe_token_unpermute=lambda x, p: x[p]),
            ),
        )
        impl = cls()
        impl.num_local_experts, impl.ep_group = 2, None
        meta = SimpleNamespace(
            input_splits=[s[rank] for s in splits],
            output_splits=splits[rank],
            reversed_global_input_permutation_mapping=perms[rank] if mapping else None,
        )
        source = sources[rank]
        before = source.clone()
        assert torch.equal(impl.exchange_combine_columns(source, meta), expected)
        assert torch.equal(source, before) and tile_index == 4


@pytest.mark.parametrize(
    "tp,rows,cap",
    [
        (2, 1, 4),
        (2, 3, 4),
        (8, 128, 1024),
        (8, 129, 1024),
        (8, 257, 2048),
        (8, 4250, 1024),
        (8, 4250, 2048),
    ],
)
def test_output_projection_matches_full_reference(tp, rows, cap):
    projection(tp, rows, cap)


@pytest.mark.parametrize("use_permutation", [True, False])
def test_combine_matches_full_reference_with_empty_ranks(use_permutation):
    moe([[2, 0, 1], [0, 0, 0], [3, 0, 2]], use_permutation)
