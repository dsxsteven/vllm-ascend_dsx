# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for opt-in behavior, without importing the NPU worker stack."""

import ast
import importlib.util
import sys
from itertools import islice
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_WORKER = _ROOT / "vllm_ascend/patch/worker"
_DENSE_FLAG = "VLLM_ASCEND_DENSE_LAYERS_SEQUENCE_PARALLEL"
_EMBEDDING_FLAG = "VLLM_ASCEND_SEQUENCE_PARALLEL_EMBEDDING"


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_patch(monkeypatch):
    def load(dense=None, embedding=None):
        for name, enabled in ((_DENSE_FLAG, dense), (_EMBEDDING_FLAG, embedding)):
            if enabled is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, str(int(enabled)))

        # Fresh upstream classes per test expose whether an import overwrites
        # existing methods, including MLP implementations installed by others.
        class DenseMLP:
            def __init__(self):
                self.down_proj = SimpleNamespace(reduce_results=True)

            def forward(self, x):
                return x + 37

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        class DecoderLayer:
            def __init__(self, vllm_config, prefix):
                config = vllm_config.model_config.hf_config
                self.mlp = DenseMLP() if config.dense else object()
                self.self_attn = SimpleNamespace(o_proj=SimpleNamespace(reduce_results=True))
                self.use_sequence_parallel_moe = not config.dense

            def forward(self, positions, hidden_states, residual, llama_4_scaling=None):
                return hidden_states + 23, residual

        class Model:
            def embed_input_ids(self, input_ids):
                return self.embedding

        original_methods = (
            DecoderLayer.__init__,
            DecoderLayer.forward,
            DenseMLP.forward,
            Model.embed_input_ids,
        )
        modules = {}
        for name in (
            "vllm",
            "vllm.envs",
            "vllm.distributed",
            "vllm.model_executor",
            "vllm.model_executor.models",
            "vllm.model_executor.models.deepseek_v2",
            "vllm.model_executor.models.utils",
            "vllm_ascend",
            "vllm_ascend.utils",
        ):
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
            monkeypatch.setitem(sys.modules, name, module)
            parent, _, child = name.rpartition(".")
            if parent in modules:
                setattr(modules[parent], child, module)
        modules["vllm.envs"].environment_variables = {}
        envs = _load_file("_dense_sp_test_envs", _ROOT / "vllm_ascend/envs.py")
        modules["vllm_ascend"].envs = envs
        distributed = modules["vllm.distributed"]
        distributed.get_tensor_model_parallel_world_size = Mock(return_value=2)
        distributed.tensor_model_parallel_all_gather = Mock(side_effect=lambda x, dim: torch.cat([x, x], dim))
        distributed.tensor_model_parallel_reduce_scatter = Mock(side_effect=lambda x, dim: x.chunk(2, dim)[0])
        upstream = modules["vllm.model_executor.models.deepseek_v2"]
        upstream.DeepseekV2MLP = DenseMLP
        upstream.DeepseekV2DecoderLayer = DecoderLayer
        upstream.DeepseekV2Model = Model
        upstream.DeepseekAttention = type("Attention", (), {})
        chunk = Mock(side_effect=lambda x: torch.nn.functional.pad(x, (0, 0, 0, -len(x) % 2)).chunk(2)[0])
        modules["vllm.model_executor.models.utils"].sequence_parallel_chunk = chunk
        modules["vllm_ascend.utils"].is_mtp_layer = lambda config, prefix: config.mtp
        patch = _load_file("_dense_sp_test_patch", _WORKER / "patch_dense_layer_sequence_parallel.py")
        return SimpleNamespace(
            patch=patch,
            envs=envs,
            upstream=upstream,
            originals=original_methods,
            distributed=distributed,
            chunk=chunk,
        )

    return load


def _config(*, sp=True, pp=1, dense=True, mtp=False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=sp, pipeline_parallel_size=pp),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(dense=dense, mtp=mtp)),
    )


@pytest.mark.parametrize("dense,embedding", [(None, None), (False, False), (False, True), (True, False), (True, True)])
def test_import_only_installs_enabled_patches(load_patch, dense, embedding):
    state = load_patch(dense, embedding)
    upstream = state.upstream
    methods = (
        upstream.DeepseekV2DecoderLayer.__init__,
        upstream.DeepseekV2DecoderLayer.forward,
        upstream.DeepseekV2MLP.forward,
        upstream.DeepseekV2Model.embed_input_ids,
    )
    assert [method is original for method, original in zip(methods, state.originals)] == [
        not dense,
        not dense,
        not dense,
        not (dense and embedding),
    ]


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("options", [{}, {"sp": False}, {"pp": 2}, {"mtp": True}, {"dense": False}])
def test_only_eligible_dense_layers_change_reductions(load_patch, dense, options):
    state = load_patch(dense, True)
    layer = state.upstream.DeepseekV2DecoderLayer(_config(**options), "model.layers.0")
    enabled = dense and not options
    assert getattr(layer.mlp, state.patch._SP_DENSE_MLP, False) is enabled
    assert layer.self_attn.o_proj.reduce_results is not enabled
    if isinstance(layer.mlp, state.upstream.DeepseekV2MLP):
        assert layer.mlp.down_proj.reduce_results is not enabled
    if not enabled:
        hidden = torch.ones(2, 4)
        residual = torch.zeros_like(hidden)
        actual, actual_residual = layer.forward(torch.arange(2), hidden, residual)
        torch.testing.assert_close(actual, hidden + 23)
        assert actual_residual is residual
        state.distributed.tensor_model_parallel_all_gather.assert_not_called()
        state.distributed.tensor_model_parallel_reduce_scatter.assert_not_called()


@pytest.mark.parametrize("dense,marked,sharded", [(False, True, True), (True, False, True), (True, True, False)])
def test_mlp_fallback_calls_original_implementation(load_patch, dense, marked, sharded):
    state = load_patch(dense, False)
    mlp = state.upstream.DeepseekV2MLP()
    setattr(mlp, state.patch._SP_DENSE_MLP, marked)
    hidden = torch.ones(2, 4)
    actual = state.patch._patched_mlp_forward(mlp, hidden, already_sequence_parallel=sharded)
    torch.testing.assert_close(actual, hidden + 37)
    state.distributed.tensor_model_parallel_all_gather.assert_not_called()
    state.distributed.tensor_model_parallel_reduce_scatter.assert_not_called()


def test_enabled_dense_mlp_gathers_and_scatters(load_patch):
    state = load_patch(True, False)
    mlp = state.upstream.DeepseekV2DecoderLayer(_config(), "model.layers.0").mlp
    mlp.gate_up_proj = lambda x: (x + 1, None)
    mlp.act_fn = lambda x: x * 2
    mlp.down_proj = lambda x: (x - 3, None)
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    actual = mlp.forward(hidden, already_sequence_parallel=True)
    torch.testing.assert_close(actual, 2 * (hidden + 1) - 3)
    state.distributed.tensor_model_parallel_all_gather.assert_called_once()
    state.distributed.tensor_model_parallel_reduce_scatter.assert_called_once()


@pytest.mark.parametrize("embedding", [False, True])
def test_enabled_first_dense_layer_accepts_full_or_sharded_embedding(load_patch, embedding):
    state = load_patch(True, embedding)
    layer = state.upstream.DeepseekV2DecoderLayer(_config(), "model.layers.0")
    layer.use_mha = True
    layer.self_attn = lambda positions, hidden: hidden * 2
    layer.input_layernorm = lambda hidden: hidden
    layer.post_attention_layernorm = lambda hidden, residual: (hidden + residual, residual)
    mlp = layer.mlp
    mlp.gate_up_proj = lambda x: (x, None)
    mlp.act_fn = lambda x: x
    mlp.down_proj = lambda x: (x, None)
    positions = torch.arange(3)
    hidden = torch.ones(2 if embedding else 3, 4)
    actual, residual = layer.forward(positions, hidden, None)
    torch.testing.assert_close(actual, torch.full((2, 4), 3.0))
    torch.testing.assert_close(residual, torch.ones(2, 4))
    # Only a sharded first-layer input requires gathering before attention.
    assert state.distributed.tensor_model_parallel_all_gather.call_count == (2 if embedding else 1)
    assert state.distributed.tensor_model_parallel_reduce_scatter.call_count == 2
    assert state.chunk.call_count == (0 if embedding else 1)


@pytest.mark.parametrize("dense,embedding", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("tokens,tp_size", [(1, 2), (2, 2), (5, 2), (5, 1)])
def test_embedding_split_requires_both_flags(load_patch, dense, embedding, tokens, tp_size):
    state = load_patch(dense, embedding)
    state.distributed.get_tensor_model_parallel_world_size.return_value = tp_size
    model = state.upstream.DeepseekV2Model()
    model.layers = [state.upstream.DeepseekV2DecoderLayer(_config(), "model.layers.0")]
    model.embedding = torch.arange(tokens * 4, dtype=torch.float32).reshape(tokens, 4)
    actual = model.embed_input_ids(torch.arange(tokens))
    if dense and embedding and tokens >= tp_size > 1:
        # The embedding is already reduced: splitting must never multiply it by TP.
        torch.testing.assert_close(actual, model.embedding[: (tokens + 1) // 2])
        state.chunk.assert_called_once()
    else:
        assert actual is model.embedding
        state.chunk.assert_not_called()
    state.distributed.tensor_model_parallel_reduce_scatter.assert_not_called()


def test_embedding_keeps_unpatched_moe_first_layer_full(load_patch):
    state = load_patch(True, True)
    model = state.upstream.DeepseekV2Model()
    model.layers = [state.upstream.DeepseekV2DecoderLayer(_config(dense=False), "model.layers.0")]
    model.embedding = torch.ones(4, 4)
    assert model.embed_input_ids(torch.arange(4)) is model.embedding
    state.chunk.assert_not_called()


@pytest.mark.parametrize("dense,embedding", [(None, None), (False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("sharded", [False, True])
def test_final_norm_order_and_values_follow_dense_flag(load_patch, dense, embedding, sharded):
    state = load_patch(dense, embedding)
    trace = []
    tokens, hidden_size = 3, 4
    rows = 2 if sharded else tokens
    hidden = torch.arange(rows * hidden_size, dtype=torch.float32).reshape(rows, hidden_size) + 1
    residual = hidden / 10

    def normalize(x, res):
        trace.append(("norm", tuple(x.shape)))
        x = x + res
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6), None

    def gather(x, dim):
        trace.append(("gather", tuple(x.shape)))
        return torch.cat([x, x], dim)

    # Execute the actual full forward and aux-capture functions. Other functions
    # in this module import hardware-specific classes irrelevant to this path.
    source = ast.parse((_WORKER / "patch_deepseek_v2.py").read_text(encoding="utf-8"))
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_patched_forward", "_capture_aux_hidden_state"}
    ]
    namespace = {
        "torch": torch,
        "islice": islice,
        "envs": state.envs,
        "IntermediateTensors": dict,
        "get_pp_group": lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        "tensor_model_parallel_all_gather": gather,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(_WORKER / "patch_deepseek_v2.py"), "exec"), namespace)
    model = SimpleNamespace(
        config=SimpleNamespace(),
        hidden_size=hidden_size,
        embed_input_ids=lambda ids: hidden,
        start_layer=0,
        end_layer=1,
        layers=[lambda *args: (hidden, residual)],
        aux_hidden_state_layers=(),
        norm=normalize,
    )
    actual = namespace["_patched_forward"](model, torch.arange(tokens), torch.arange(tokens), None)
    if sharded:
        if dense:
            assert trace == [("norm", (rows, hidden_size)), ("gather", (rows, hidden_size))]
        else:
            assert trace == [("gather", (rows, 2 * hidden_size)), ("norm", (tokens, hidden_size))]
        full = torch.cat([hidden + residual, hidden + residual])[:tokens]
    else:
        assert trace == [("norm", (tokens, hidden_size))]
        full = hidden + residual
    expected = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name", [_DENSE_FLAG, _EMBEDDING_FLAG])
@pytest.mark.parametrize(
    "value,expected", [(None, False), ("0", False), ("1", True), ("", None), ("2", None), ("true", None)]
)
def test_switches_are_strict_and_default_off(load_patch, monkeypatch, name, value, expected):
    state = load_patch()
    if value is not None:
        monkeypatch.setenv(name, value)
    if expected is None:
        with pytest.raises(ValueError):
            getattr(state.envs, name)
    else:
        assert getattr(state.envs, name) is expected
