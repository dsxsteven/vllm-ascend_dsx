# SPDX-License-Identifier: Apache-2.0
"""CPU control-flow checks; native HC input immutability is checked separately."""

import ast
import copy
from pathlib import Path

import pytest
import torch

from tests.ut.models.test_deepseek_v4_residual_lifetime import _cpu_decoder, _decoder_namespace, _load_forward


def _clone_reference():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v4/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4DecoderLayer")
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
    replacements = 0
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "residual" for t in node.targets):
            node.value = ast.parse("hidden_states.clone()", mode="eval").body
            replacements += 1
    assert replacements == 2
    ns = _decoder_namespace()
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns["forward"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "offset", "strided"])
@pytest.mark.parametrize("use_sp,use_cp", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("profile", [False, True])
def test_residual_alias_matches_clone_and_preserves_input(dtype, layout, use_sp, use_cp, profile):
    data = torch.arange(128, dtype=dtype).reshape(8, 4, 4) / 32
    hidden = data[:4] if layout == "contiguous" else data[1:5] if layout == "offset" else data[::2]
    before = hidden.clone()
    positions = torch.arange(8 if use_sp and not use_cp else 4)
    candidate = _load_forward("DeepseekV4DecoderLayer", _decoder_namespace())
    clone = _clone_reference()

    def run(forward, check_alias):
        layer = _cpu_decoder(use_sp, profile)
        layer.use_sequence_parallel_moe = use_sp
        layer.enable_dsa_cp = use_cp
        pending = []
        aliases = []

        def pre(x, *_):
            pending.append((x.data_ptr(), x.is_contiguous(), x.clone()))
            # Later stages can mutate this NEW output without touching x.
            return x.mean(dim=1), None, None

        def post(x, residual, *_):
            ptr, contiguous, values = pending.pop(0)
            torch.testing.assert_close(residual, values, rtol=0, atol=0)
            aliases.append(ptr == residual.data_ptr())
            if check_alias:
                assert aliases[-1] == contiguous
            else:
                assert not aliases[-1]
            return (x.unsqueeze(1) + residual).contiguous()

        layer.hc_pre, layer.hc_post = pre, post
        layer.input_layernorm = lambda x: x.add_(1)
        # Emulate the expanded attention rows for non-CP sequence parallel.
        layer.self_attn = lambda *, hidden_states, **_: hidden_states + 1
        output = forward(layer, positions, hidden, None)
        assert len(aliases) == 2 and not pending
        torch.testing.assert_close(hidden, before, rtol=0, atol=0)
        return output

    with torch.inference_mode():
        expected = run(clone, False)
        actual = run(candidate, True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
