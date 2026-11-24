# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout

from tests.ut.distributed.ascend_store.test_layerwise_cache_layout import _native_config, _runtime
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    LayerwiseCachePlanner,
    get_spec_reuse_rules,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_cache_views,
    get_layerwise_reuse_backing_bytes,
)
from vllm_ascend.models.glm_moe_dsa_cache_reuse import normalize_offload_descriptors


def _glm_runtime(num_buffers=3, model_type="glm_moe_dsa"):
    runtime = _runtime(78, num_buffers, model_type=model_type)
    runtime.model_config.hf_config = runtime.model_config.hf_text_config
    runtime.cache_config.get_resolved_kv_cache_layout = lambda: KVCacheLayout.LBNHC
    runtime.cache_config.prefix_cache_retention_interval = 0
    runtime.attention_config = SimpleNamespace(hisparse_config=None)
    runtime.speculative_config = None
    return runtime


def _glm_specs(c8=True, query_heads=64):
    mla = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        num_heads=query_heads,
        head_size=656 if c8 else 576,
        dtype=torch.int8 if c8 else torch.bfloat16,
        cache_sparse_sfa_c8=c8,
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8 if c8 else torch.bfloat16,
        scale_dim=1 if c8 else 0,
        scale_dtype=torch.float32,
        cache_sparse_li_c8=c8,
    )
    specs = {f"model.layers.{i}.self_attn.attn": mla for i in range(78)}
    specs.update({f"model.layers.{i}.indexer.k_cache": indexer for i in range(21)})
    return mla, indexer, specs


def _glm_group(specs):
    return KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs))


def _builder():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    return AscendKVCacheConfigBuilder()


@pytest.mark.parametrize("blocks", [1, 3, 7881])
def test_glm5_full_layer_reuse_budget(blocks):
    mla, indexer, _ = _glm_specs()
    groups = [
        KVCacheGroupSpec([f"model.layers.{i}.self_attn" for i in range(78)], mla),
        KVCacheGroupSpec([f"model.layers.{i}.indexer" for i in range(0, 78, 4)], indexer),
    ]
    native = _native_config(groups, blocks)
    runtime = _glm_runtime()
    layout = plan_layerwise_cache_reuse(native, runtime)
    assert layout is not None and layout.has_layer_reuse
    assert "model.layers.0.self_attn" not in layout.reused_cache_names
    assert "model.layers.0.indexer" not in layout.reused_cache_names
    assert all(f"model.layers.{i}.self_attn" in layout.reused_cache_names for i in range(1, 78))
    assert layout.kv_cache_tensors[0].size == 4 * blocks * (mla.page_size_bytes + indexer.page_size_bytes)
    assert layout.kv_cache_tensors[0].size < sum(t.size for t in native.kv_cache_tensors)
    assert all(source < layer for layer, sources in layout.prefetch_layer_map.items() for source in sources)
    runtime.kv_transfer_config.kv_connector_extra_config.pop("layerwise_num_shared_buffers")
    assert plan_layerwise_cache_reuse(native, runtime) is None


def test_glm5_rules_do_not_enable_unknown_specs_or_models():
    rules = get_spec_reuse_rules(_glm_runtime())
    dense = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    assert not LayerwiseCachePlanner.can_reuse_spec(dense, rules)
    assert get_spec_reuse_rules(_glm_runtime(model_type="unknown")) is None


@pytest.mark.parametrize("blocks", [1, 3, 7881])
@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("query_heads", [1, 64])
def test_real_builder_uses_physical_mla_and_li_pages(blocks, c8, query_heads):
    mla, indexer, specs = _glm_specs(c8, query_heads)
    runtime = _glm_runtime()
    builder = _builder()
    groups = [_glm_group(specs)]
    cost = builder._get_pool_bytes_per_block(runtime, groups)
    # Layer 0 is resident; the other layers rotate through three slots.
    assert cost == 4 * (mla.page_size_bytes + indexer.page_size_bytes)
    config = builder.get_kv_cache_config_from_groups(runtime, groups, blocks)
    assert get_layerwise_reuse_backing_bytes(config) == cost * blocks
    assert all(t.block_stride == specs[n].page_size_bytes for t in config.kv_cache_tensors for n in t.layers)
    assert mla.num_heads == query_heads


def _native_glm_config():
    _, _, specs = _glm_specs()
    runtime = _glm_runtime(num_buffers=None)
    config = _builder().get_kv_cache_config_from_groups(runtime, [_glm_group(specs)], 3)
    return runtime, config, specs


def test_normalization_preserves_query_heads_and_is_idempotent():
    runtime, config, specs = _native_glm_config()
    normalize_offload_descriptors(runtime, config, specs)
    tensors = config.kv_cache_tensors
    assert set(get_cache_views(config, specs)) == set(specs)
    assert specs["model.layers.0.self_attn.attn"].num_heads == 64
    normalize_offload_descriptors(runtime, config, specs)
    assert config.kv_cache_tensors is tensors


@pytest.mark.parametrize("mode", ["disabled", "unknown_model", "other_layout", "empty"])
def test_non_offload_and_unsupported_layouts_are_unchanged(mode):
    runtime, config, specs = _native_glm_config()
    if mode == "unknown_model":
        runtime = _glm_runtime(model_type="unknown")
    elif mode == "other_layout":
        runtime.cache_config.get_resolved_kv_cache_layout = lambda: SimpleNamespace(name="NBLHC")
    elif mode == "empty":
        config.num_blocks = 0
        config.kv_cache_tensors = []
    tensors = config.kv_cache_tensors
    if mode in ("other_layout", "empty"):
        normalize_offload_descriptors(runtime, config, specs)
    else:
        assert plan_layerwise_cache_reuse(config, runtime) is None
    assert config.kv_cache_tensors is tensors


@pytest.mark.parametrize("malformation", ["duplicate", "missing", "empty_layers", "size", "stride"])
def test_unrecognized_native_descriptors_fail_without_partial_mutation(malformation):
    runtime, config, specs = _native_glm_config()
    tensor = config.kv_cache_tensors[0]
    if malformation == "duplicate":
        config.kv_cache_tensors.append(tensor)
    elif malformation == "missing":
        config.kv_cache_tensors.pop()
    elif malformation == "empty_layers":
        config.kv_cache_tensors[0] = replace(tensor, layers=[])
    elif malformation == "size":
        config.kv_cache_tensors[-1] = replace(config.kv_cache_tensors[-1], size=1)
    else:
        config.kv_cache_tensors[-1] = replace(config.kv_cache_tensors[-1], block_stride=7)
    tensors = config.kv_cache_tensors
    with pytest.raises(ValueError, match="GLM-5 offload"):
        normalize_offload_descriptors(runtime, config, specs)
    assert config.kv_cache_tensors is tensors


def _publication_worker(monkeypatch, tp, pcp, offload=True, transfer=True):
    events = []
    monkeypatch.setattr(pool_worker, "get_tp_group", lambda: SimpleNamespace(barrier=lambda: events.append("tp")))
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(barrier=lambda: events.append("pcp")))
    monkeypatch.setattr(pool_worker, "reset_attention_compute_start_gate", lambda: None)
    worker = pool_worker.LayerwiseKVPoolWorker.__new__(pool_worker.LayerwiseKVPoolWorker)
    worker.use_layerwise = True
    worker.use_layerwise_transfer = transfer
    worker.layerwise_offload = offload
    worker.tp_size, worker.pcp_size = tp, pcp
    worker.use_kvpp = False
    worker.kv_recv_thread = None
    worker._save_disabled = set()
    worker._drain_deferred_last_save = lambda: events.append("save_drained")
    worker.process_layer_data = lambda requests: events.append("load_prepared")
    return worker, events


@pytest.mark.parametrize(
    ("tp", "pcp", "offload", "transfer"),
    [
        (1, 8, True, True),
        (2, 1, True, True),
        (2, 8, True, True),
        (1, 1, True, True),
        (1, 8, False, True),
        (1, 8, True, False),
    ],
)
def test_previous_step_publication_fenced_before_pcp_load(monkeypatch, tp, pcp, offload, transfer):
    worker, events = _publication_worker(monkeypatch, tp, pcp, offload, transfer)
    worker.prepare_layerwise_step(SimpleNamespace(requests=[]))
    expected = ["save_drained"]
    if transfer and offload:
        if tp > 1:
            expected.append("tp")
        if pcp > 1:
            expected.append("pcp")
    assert events == expected + ["load_prepared"]


def test_failed_publication_fence_does_not_prepare_loads(monkeypatch):
    worker, events = _publication_worker(monkeypatch, 1, 8)

    def fail():
        raise RuntimeError("publication failed")

    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(barrier=fail))
    with pytest.raises(RuntimeError, match="publication failed"):
        worker.prepare_layerwise_step(SimpleNamespace(requests=[]))
    assert events == ["save_drained"]
