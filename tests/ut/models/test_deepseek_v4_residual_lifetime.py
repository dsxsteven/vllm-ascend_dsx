# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks of production control flow; NPU kernels/collectives are stubbed.

Load only forward methods to avoid requiring the Ascend runtime. In particular,
do not use call-recording mocks for tensors: they would keep residuals alive.
"""

import __future__

import ast
import weakref
from contextlib import contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_forward(class_name, namespace):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v4/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["forward"]


def _cpu_decoder(use_sp=False, profile=False):
    return SimpleNamespace(
        activation_peak_debug=profile,
        activation_peak_phase="target",
        layer_idx=0,
        use_sequence_parallel_moe=use_sp,
        enable_dsa_cp=False,
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        hc_ffn_fn=None,
        hc_ffn_scale=None,
        hc_ffn_base=None,
        hc_pre=lambda x, *_: (x.mean(dim=1), None, None),
        hc_post=lambda x, residual, *_: x.unsqueeze(1) + residual,
        input_layernorm=lambda x: x,
        rms_norm_cast=lambda x: (x, x.float()),
        self_attn=lambda *, hidden_states, **_: hidden_states + 1,
        mlp=lambda x, **_: x * 2,
    )


def _decoder_namespace():
    return {
        "torch": torch,
        "nullcontext": nullcontext,
        "is_activation_peak_profiling": lambda: True,
        "record_activation_peak": lambda *a, **kw: nullcontext(),
        "record_activation_substage": lambda *a, **kw: nullcontext(),
        # Two identical token shards; this only tests SP control flow.
        "sp_all_gather": lambda x: torch.cat((x, x)),
        "sp_reduce_scatter": lambda x: x.chunk(2)[0],
    }


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoder_does_not_consume_incoming_residual(profile, dtype):
    forward = _load_forward("DeepseekV4DecoderLayer", _decoder_namespace())
    layer = _cpu_decoder(profile=profile)
    hidden = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
    positions = torch.arange(2)
    with torch.inference_mode():
        expected, expected_residual = forward(layer, positions, hidden, torch.full_like(hidden, float("nan")))
        actual, actual_residual = forward(layer, positions, hidden, None)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)


@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("with_aux", [False, True])
@pytest.mark.parametrize(
    "first,last,needs_mtp",
    [(True, True, False), (True, True, True), (True, False, False), (False, True, True), (False, False, False)],
)
def test_model_releases_residual_before_next_layer_and_tail(use_sp, with_aux, first, last, needs_mtp):
    residual_refs = []
    decoder_forward = _load_forward("DeepseekV4DecoderLayer", _decoder_namespace())

    def assert_released():
        assert all(ref() is None for ref in residual_refs), "unused residual retained by the model loop"

    class Layer:
        def __init__(self, index):
            self.layer_idx = index
            self.decoder = _cpu_decoder(use_sp)

        def __call__(self, positions, hidden, residual, scaling, *, input_ids):
            assert_released()
            output, next_residual = decoder_forward(
                self.decoder, positions, hidden, residual, scaling, input_ids=input_ids
            )
            residual_refs.append(weakref.ref(next_residual))
            return output, next_residual

    @contextmanager
    def peak_scope(kind, *args, **kwargs):
        if kind == "target_tail":
            assert_released()
        yield
        if kind == "target_layer":
            # Also covers release before aux outputs and after the final layer.
            assert_released()

    def substage_scope(name):
        if name == "mtp_hidden_stash":
            assert_released()
        return nullcontext()

    def add_transport(tensors, kind, auxiliary):
        assert_released()
        tensors[kind] = auxiliary
        return tensors

    namespace = {
        "torch": torch,
        "islice": islice,
        "get_pp_group": lambda: SimpleNamespace(is_first_rank=first, is_last_rank=last),
        "IntermediateTensors": dict,
        "PPTransportDataType": SimpleNamespace(AUX_HIDDEN_STATES="aux"),
        "get_pp_transport_tensors": lambda tensors, kind: list(tensors.get(kind, [])) if tensors else [],
        "add_pp_transport_tensors": add_transport,
        "record_activation_peak": peak_scope,
        "record_activation_substage": substage_scope,
        "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
        "sp_shard": lambda x: x.chunk(2)[0],
        "sp_all_gather": lambda x: torch.cat((x, x)),
    }
    forward = _load_forward("DeepseekV4Model", namespace)
    embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4).repeat(2, 1)
    expanded = embeddings.unsqueeze(1).repeat(1, 4, 1)
    upstream_aux = [embeddings - 1] if with_aux and not first else []
    intermediate = None if first else {"hidden_states": expanded, "aux": upstream_aux}
    positions = torch.arange(6)
    input_ids = torch.arange(6) if first else None
    model = SimpleNamespace(
        layers=[Layer(i) for i in range(3)],
        start_layer=0,
        end_layer=3,
        use_sequence_parallel_moe=use_sp,
        hc_mult=4,
        embed_input_ids=lambda _: embeddings,
        aux_hidden_state_layers=[1, 3] if with_aux else [],
        _needs_mtp_hidden_states=needs_mtp,
        _mtp_hidden_buffer=None,
        _mtp_buffer_shape=(6, 16),
        _mtp_buffer_dtype=torch.float32,
        device="cpu",
        hc_head_fn=None,
        hc_head_scale=None,
        hc_head_base=None,
        hc_head=lambda x, *_: x.mean(dim=1),
        norm=lambda x: x + 5,
    )
    expected_hidden = expanded
    expected_aux = list(upstream_aux)
    for index in range(3):
        after_attn = expected_hidden + (expected_hidden.mean(dim=1) + 1).unsqueeze(1)
        expected_hidden = after_attn + (after_attn.mean(dim=1) * 2).unsqueeze(1)
        if index + 1 in model.aux_hidden_state_layers:
            expected_aux.append(expected_hidden.mean(dim=1))

    # A second call covers reusing the MTP buffer and both embedding inputs.
    for use_embeds in (False, True):
        with torch.inference_mode():
            result = forward(model, input_ids, positions, intermediate, embeddings if first and use_embeds else None)
        assert_released()
        if last:
            output, auxiliary = result if expected_aux else (result, [])
            torch.testing.assert_close(output, expected_hidden.mean(dim=1) + 5, rtol=0, atol=0)
            if needs_mtp:
                torch.testing.assert_close(model._mtp_hidden_buffer, expected_hidden.flatten(1), rtol=0, atol=0)
        else:
            torch.testing.assert_close(result["hidden_states"], expected_hidden, rtol=0, atol=0)
            auxiliary = result["aux"]
        assert len(auxiliary) == len(expected_aux)
        for actual, expected in zip(auxiliary, expected_aux):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(residual_refs) == 6
