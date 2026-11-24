# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5 MLA and indexer cache eligibility for prefill buffer reuse."""

from dataclasses import replace
from types import MappingProxyType

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, KVCacheTensor, MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules

GLM_MOE_DSA_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
    {
        AscendMLAAttentionSpec: True,
        AscendSFAIndexerCacheSpec: True,
        MLAAttentionSpec: True,
    }
)


def normalize_offload_descriptors(
    vllm_config: VllmConfig,
    cache: KVCacheConfig,
    specs: dict[str, KVCacheSpec],
) -> None:
    """Restore physical MLA/LI strides before planning GLM-5 buffer reuse.

    Ascend MLA's ``num_heads`` carries query-head metadata for kernel selection.
    The generic LBNHC planner instead treats it as stored KV heads, inflating
    strides and subsequent offsets without increasing the backing allocation.
    A GLM-5 UniformType group stores one latent MLA page and separate indexer
    pages. Repack its descriptors using their physical byte sizes; retain the
    query-head metadata and do not change non-offload or other model layouts.
    """
    if vllm_config.cache_config.get_resolved_kv_cache_layout().name != "LBNHC" or len(cache.kv_cache_groups) != 1:
        return
    tensors = cache.kv_cache_tensors
    if not tensors or cache.num_blocks == 0:
        return
    names = [name for tensor in tensors for name in tensor.layers]
    if any(not tensor.layers for tensor in tensors) or len(names) != len(set(names)) or set(names) != set(specs):
        raise ValueError("GLM-5 offload descriptors must cover every cache entry exactly once")
    if not all(type(spec) in (AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec) for spec in specs.values()):
        return
    if all(tensor.block_stride == specs[name].page_size_bytes for tensor in tensors for name in tensor.layers):
        return

    total = cache.num_blocks * sum(spec.page_size_bytes for spec in specs.values())
    normalized: list[KVCacheTensor] = []
    offset = 0
    for tensor in tensors:
        spec = specs[tensor.layers[0]]
        page = spec.page_size_bytes
        logical = spec.num_heads * spec.get_num_kernel_states(spec.block_size) * spec.state_content_size_bytes
        if tensor.size != total or tensor.block_stride not in (page, logical):
            raise ValueError("Unrecognized GLM-5 offload descriptor geometry")
        if any(specs[name].page_size_bytes != page for name in tensor.layers):
            raise ValueError("GLM-5 offload descriptor mixes physical page sizes")
        normalized.append(replace(tensor, offset=offset, block_stride=page, layer_stride=cache.num_blocks * page))
        offset += len(tensor.layers) * cache.num_blocks * page
    if offset != total:
        raise ValueError("GLM-5 offload descriptors do not cover their physical backing")
    # Publish only after validating every descriptor, so errors leave it intact.
    cache.kv_cache_tensors = normalized
