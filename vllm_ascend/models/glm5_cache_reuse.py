# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM sparse-attention cache eligibility for layerwise prefill offload."""

from types import MappingProxyType

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules

# Both the main MLA KV and the indexer keys are saved and restored per layer.
# Their native byte strides remain distinct; reuse fences cover both entries.
# The planner keeps independent layers and speculative draft caches resident.
GLM5_SFA_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
    {
        AscendMLAAttentionSpec: True,
        AscendSFAIndexerCacheSpec: True,
    }
)
