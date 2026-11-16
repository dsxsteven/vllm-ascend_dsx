# SPDX-License-Identifier: Apache-2.0
"""Execute rebased entry points on CPU; NPU kernels are explicit mock boundaries."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend"


def load_forward(file, cls_name, namespace):
    path = ROOT / file
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name)
    cls.bases, cls.decorator_list = [], []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward"]
    assert len(cls.body) == 1
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls]
    namespace["torch"] = torch
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[cls_name]()


@pytest.mark.parametrize("profiling", [False, True])
@pytest.mark.parametrize("tokens", [None, 0, 3])
def test_dsa_v41_prepares_caches_once_and_reuses_output(profiling, tokens):
    metadata = SimpleNamespace(
        swa=SimpleNamespace(num_actual_tokens=tokens), positions=torch.arange(3), rope=Mock(return_value=(1, 2))
    )
    scopes = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    impl = load_forward(
        "attention/dsa_v41.py",
        "AscendDSAV41Impl",
        dict(
            get_forward_context=lambda: SimpleNamespace(attn_metadata=metadata if tokens is not None else None),
            is_activation_peak_profiling=lambda: profiling,
            record_activation_substage=scopes,
            log_activation_skip=Mock(),
            nullcontext=nullcontext,
        ),
    )
    impl.prefix = "layer.0"
    impl._get_layer_metadata = Mock(return_value=metadata)
    impl._prepare_inputs_and_caches = Mock(return_value=("q", "qr", "prepared_indexer"))
    impl._select_sparse_indices = Mock(return_value="indices")
    impl._forward_attention = Mock(return_value=torch.ones(3, 2, 4))
    type(impl)._apply_rotary = Mock()
    impl._project_output = Mock(side_effect=lambda *args, projected: projected.fill_(7))
    attn = SimpleNamespace(rotary_emb=SimpleNamespace(layername="rope"), n_local_heads=2, head_dim=4)
    hidden = torch.ones(3, 8)
    output = torch.full_like(hidden, -1)
    assert impl.forward(attn, torch.arange(3), hidden, output) is output
    if tokens is None:
        impl._prepare_inputs_and_caches.assert_not_called()
        impl._project_output.assert_not_called()
        assert torch.count_nonzero(output) == 0
        return
    impl._prepare_inputs_and_caches.assert_called_once_with(attn, hidden, metadata, metadata)
    impl._project_output.assert_called_once()
    assert torch.all(output == 7)
    if tokens:
        impl._select_sparse_indices.assert_called_once()
        assert impl._select_sparse_indices.call_args.args[-1] == "prepared_indexer"
        impl._forward_attention.assert_called_once_with(attn, "q", metadata, "indices")
        type(impl)._apply_rotary.assert_called_once()
    else:
        impl._select_sparse_indices.assert_not_called()
        impl._forward_attention.assert_not_called()
        assert impl._project_output.call_args.args[1].shape == (0, 2, 4)
    assert scopes.call_count == ((4 if tokens else 2) if profiling else 0)


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_sfa_full_token_chunk_plan_does_not_consume_sequence_parallel_shards(sequence_parallel):
    impl = load_forward("attention/sfa_v1.py", "AscendSFAImpl", dict(PreprocessType=SimpleNamespace(MLAPO="mlapo")))
    impl.enable_sparse_sfa_turboquant = False
    impl.qk_rope_head_dim = 64
    impl.g_proj = None
    impl.runtime_has_indexer = False
    impl.preprocess_type = "native"
    impl._compose_sfa_kv_cache = lambda cache: cache
    impl._get_sfa_kv_slot_mapping = Mock()
    impl._get_indexer_attn_metadata = Mock(return_value=None)
    output = torch.empty(3, 8)
    hidden = torch.ones_like(output)
    impl._forward_attention_in_chunks = Mock(return_value=output)

    class NormalPathReached(Exception):
        pass

    impl._get_parallel_forward_context = Mock(side_effect=NormalPathReached)
    metadata = SimpleNamespace(cos=1, sin=2, chunks=SimpleNamespace(chunk_size=2), num_input_tokens=3)
    args = ("layer.0", hidden, (torch.empty(1),), metadata, output)
    if sequence_parallel:
        with pytest.raises(NormalPathReached):
            impl.forward(*args, input_is_sequence_parallel=True, output_scale=0.5)
        impl._forward_attention_in_chunks.assert_not_called()
        impl._get_parallel_forward_context.assert_called_once_with(metadata, 3, hidden)
    else:
        assert impl.forward(*args) is output
        impl._forward_attention_in_chunks.assert_called_once()
        impl._get_parallel_forward_context.assert_not_called()
