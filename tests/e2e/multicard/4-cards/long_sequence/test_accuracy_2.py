#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#
"""
Compare the outputs of vLLM with and without context parallel.

Run `pytest tests/e2e/multicard/long_sequence/test_accuracy.py`.
"""

import pytest,os

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "8,9,10,11"

# MODELS = [
#     "/mnt/share/d00933242/DeepSeek-V2-Lite-w8a8",
# ]
model = "/mnt/share/d00933242/DeepSeek-V2-Lite-w8a8"

# model = "/mnt/weight/Qwen3-14B"

@wait_until_npu_memory_free(target_free_percentage=0.6)
@pytest.mark.parametrize("max_tokens", [10])
def test_models_long_sequence_cp_kv_interleave_size_output_between_tp_and_cp(
    max_tokens: int,
) -> None:
    prompts = [
        "The president of the United States is"
    ]

    GOLDEN_TEXT_DS = 'The president of the United States is a man who has been elected to the highest office'
    GOLDEN_TOKENS_DS = [100000, 549, 6847, 280, 254, 4794, 5110, 317, 245, 668, 779, 643, 803, 19136, 276, 254, 7492, 4995]

    GOLDEN_TEXT_QWEN = 'The president of the United States is the chief executive of the federal government. The president'
    GOLDEN_TOKENS_QWEN = [785, 4767, 315, 279, 3639, 4180, 374, 279, 10178, 10905, 315, 279, 6775, 3033, 13, 576, 4767]

    GOLDEN_DS = [(GOLDEN_TOKENS_DS,GOLDEN_TEXT_DS)]
    GOLDEN_QWEN = [(GOLDEN_TOKENS_QWEN,GOLDEN_TEXT_QWEN)]
    
    common_kwargs = {
        "max_model_len": 1024,
    }

    if model == "/mnt/share/d00933242/DeepSeek-V2-Lite-w8a8":
        cp_kwargs = {
            "tensor_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 2,
            "enable_expert_parallel": True,
            "cp_kv_cache_interleave_size": 128,
            "enforce_eager": True,
            "quantization": "ascend",
        }

        cp_full_kwargs = {}
        cp_full_kwargs.update(common_kwargs)  # type: ignore
        cp_full_kwargs.update(cp_kwargs)  # type: ignore

        with VllmRunner(model, **cp_full_kwargs) as runner:  # type: ignore
            vllm_context_parallel_outputs = runner.generate_greedy(
                prompts, max_tokens)

        check_outputs_equal(
            outputs_0_lst=GOLDEN_DS,
            outputs_1_lst=vllm_context_parallel_outputs,
            name_0="GOLDEN_DS",
            name_1="vllm_context_parallel_outputs",
        )

    else:
        cp_kwargs = {
            "tensor_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 2,
            "cp_kv_cache_interleave_size": 128,
            "enforce_eager": True,
        }

        cp_full_kwargs = {}
        cp_full_kwargs.update(common_kwargs)  # type: ignore
        cp_full_kwargs.update(cp_kwargs)  # type: ignore

        with VllmRunner(model, **cp_full_kwargs) as runner:  # type: ignore
            vllm_context_parallel_outputs = runner.generate_greedy(
                prompts, max_tokens)

        check_outputs_equal(
            outputs_0_lst=GOLDEN_QWEN,
            outputs_1_lst=vllm_context_parallel_outputs,
            name_0="GOLDEN_QWEN",
            name_1="vllm_context_parallel_outputs",
    )