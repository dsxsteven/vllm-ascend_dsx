#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


def get_pcp_profile_num_tokens(vllm_config: VllmConfig, max_num_tokens: int, max_num_reqs: int) -> int:
    """Keep replicated draft requests when profiling PCP-local prefill.

    Initial memory profiling precedes KV/PCP-manager initialization.
    Block drafters retain the global budget: their context-KV precompute
    processes all target hidden states, independently of short queries.
    """
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        return max_num_tokens
    query_per_req = 1
    spec = vllm_config.speculative_config
    if spec is not None:
        # Use shared config, not the last PP rank's speculator instance:
        # every pipeline stage must profile the same target token shape.
        if spec.method not in ("mtp", "eagle", "eagle3"):
            return max_num_tokens
        query_per_req = 1 + spec.num_speculative_tokens
    local_tokens = (max_num_tokens + pcp_size - 1) // pcp_size
    # DualChunkSwap rounds two chunks per request on each PCP rank.
    padding_tokens = 2 * max_num_reqs
    profile_tokens = min(max_num_tokens, max(local_tokens, max_num_reqs * query_per_req) + padding_tokens)

    from vllm.config import CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    if not vllm_config.model_config.enforce_eager and compilation_config.cudagraph_mode in (
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ):
        # Capture sizes are already PCP-local, including graph padding.
        # Older vLLM versions use the shared token grid instead of a prefill grid.
        prefill_sizes = getattr(compilation_config, "prefill_cudagraph_capture_token_sizes", None)
        if prefill_sizes is None:
            prefill_sizes = compilation_config.cudagraph_capture_sizes
        profile_tokens = max(profile_tokens, max(prefill_sizes or [], default=0))
    return profile_tokens
