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
# This file is a part of the vllm-ascend project.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import torch
from vllm.distributed import get_dp_group
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.activation_memory import is_activation_peak_profiling, log_ffn_chunk_decision
from vllm_ascend.ascend_config import get_ascend_config, is_mega_moe_supported
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import (
    MoEFusedExpertsInput,
    MoEWeights,
    slice_fused_experts_input_along_tokens,
)
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput, build_mlp_compute_input
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import build_token_dispatch_input
from vllm_ascend.ops.fused_moe.moe_ffn_chunking import (
    balanced_chunk_ranges,
    run_moe_ffn_in_token_chunks,
    supports_moe_ffn_chunking,
)
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
)
from vllm_ascend.quantization.quant_type import QuantType

_MoECommMethods: dict[tuple[MoECommType | None, tuple[int, ...]], MoECommMethod] = {}


def _moe_config_key(
    moe_comm_type: MoECommType | None, moe_config: FusedMoEConfig | None
) -> tuple[MoECommType | None, tuple[int, ...]]:
    """Return the execution shape that owns mutable MoE comm state."""
    _CONFIG_KEY_FIELDS = ("num_experts", "num_local_experts")
    return (moe_comm_type, tuple(int(getattr(moe_config, field, 0) or 0) for field in _CONFIG_KEY_FIELDS))


def get_moe_comm_method(
    moe_comm_type: MoECommType | None,
    moe_config: FusedMoEConfig | None = None,
) -> MoECommMethod | None:
    return _MoECommMethods.get(_moe_config_key(moe_comm_type, moe_config))


def setup_moe_comm_method(moe_config):
    if moe_config.ep_size > 1:
        _MoECommMethods[_moe_config_key(MoECommType.ALLTOALL, moe_config)] = AlltoAllCommImpl(moe_config)
        _MoECommMethods[_moe_config_key(MoECommType.ALLGATHER, moe_config)] = AllGatherCommImpl(moe_config)
        _MoECommMethods[_moe_config_key(MoECommType.MC2, moe_config)] = MC2CommImpl(moe_config)
        _MoECommMethods[_moe_config_key(MoECommType.FUSED_MC2, moe_config)] = FusedMC2CommImpl(moe_config)
    else:
        _MoECommMethods[_moe_config_key(MoECommType.ALLGATHER, moe_config)] = AllGatherCommImpl(moe_config)


def _e2e_uniform_chunk_count(num_tokens: int, chunk_size: int, moe_config: FusedMoEConfig) -> int:
    """Chunk count every rank in the EP group can use (0 disables chunking).

    A chunked forward issues one dispatch/combine collective per chunk, so
    all ranks must agree on the NUMBER of chunks. Chunk SIZES may differ:
    the AllToAllV dispatcher already tolerates uneven per-rank token splits
    (``input_splits``/``output_splits``), and TP ranks always share the
    batch. We take one MIN/MAX all-reduce over the DP group and compute

        K = min(max_local_ceil(num_tokens / chunk_size), global_min_tokens)

    so every rank can produce K non-empty balanced chunks. K == 1 (or an
    idle rank with 0 tokens) disables chunking everywhere, keeping the
    collective call count identical on the fallback path.
    """
    k_local = max(1, (num_tokens + chunk_size - 1) // chunk_size)
    if getattr(moe_config, "dp_size", 1) <= 1:
        return k_local if k_local > 1 else 0
    dp_group = get_dp_group()
    if dp_group.world_size <= 1:
        return k_local if k_local > 1 else 0
    if get_ascend_config().dp_allreduce_on_npu:
        payload = torch.tensor([k_local, -num_tokens], dtype=torch.int64, device="npu")
        group = dp_group.device_group
    else:
        payload = torch.tensor([k_local, -num_tokens], dtype=torch.int64)
        group = dp_group.cpu_group
    torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.MAX, group=group)
    k_uniform = min(int(payload[0].item()), int(-payload[1].item()))
    return k_uniform if k_uniform > 1 else 0


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.lora_context = None

        ascend_config = get_ascend_config()
        self.enable_ffn_chunking = getattr(ascend_config, "enable_ffn_chunking", False) is True
        self.ffn_chunk_size = ascend_config.ffn_chunk_size

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context
        self.prepare_finalize.set_lora_context(lora_context)
        self.token_dispatcher.set_lora_context(lora_context)

    def _log_chunk_decision(
        self,
        *,
        path: str,
        raw_tokens: int,
        dispatched_tokens: int | None,
        num_chunks: int,
        supported: bool = True,
        fallback_reason: str | None = None,
    ) -> None:
        applied = self.enable_ffn_chunking and supported and num_chunks > 1
        if applied:
            reason = "chunked"
        elif not self.enable_ffn_chunking:
            reason = "disabled"
        elif not supported:
            reason = "unsupported_payload"
        elif fallback_reason is not None:
            reason = fallback_reason
        else:
            reason = "token_count_le_chunk_size"

        log_ffn_chunk_decision(
            comm_method=type(self).__name__,
            path=path,
            enabled=self.enable_ffn_chunking,
            raw_tokens=raw_tokens,
            dispatched_tokens=dispatched_tokens,
            chunk_size=self.ffn_chunk_size,
            num_chunks=max(1, num_chunks),
            applied=applied,
            reason=reason,
        )

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=replace_allreduce,
            quant_type=quant_type,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        return hidden_states

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        quant_method=None,
    ):
        # Check constraints
        assert fused_experts_input.hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.float8_e4m3fn,
            torch.uint8,
        ], f"Unsupported hidden_states dtype: {fused_experts_input.hidden_states.dtype}"

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        before_dispatch_evt = torch.npu.current_stream().record_event()

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            moe_config=self.moe_config,
        )
        if is_activation_peak_profiling():
            chunking_supported = supports_moe_ffn_chunking(mlp_compute_input)
            num_chunks = self._estimate_ffn_num_chunks(mlp_compute_input) if self.enable_ffn_chunking else 1
            self._log_chunk_decision(
                path="MLP_ONLY_AFTER_DISPATCH",
                raw_tokens=fused_experts_input.hidden_states.shape[0],
                dispatched_tokens=mlp_compute_input.hidden_states.shape[0],
                num_chunks=num_chunks,
                supported=chunking_supported,
            )

        if self.enable_ffn_chunking:
            mlp_output, before_gmm2_evt = self._apply_mlp_with_optional_chunking(mlp_compute_input, quant_method)
        else:
            mlp_output, before_gmm2_evt = self._apply_mlp_dispatched(mlp_compute_input, quant_method)

        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )

        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
        )

    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> tuple[torch.Tensor, object | None]:
        raise NotImplementedError(
            "Comm method must either override _apply_mlp or receive a quant_method "
            " so the MLP stage can be orchestrated by MoeActionMethod."
        )

    def _apply_mlp_dispatched(
        self,
        mlp_compute_input: MoEMlpComputeInput,
        quant_method,
    ) -> tuple[torch.Tensor, object | None]:
        """Apply the MLP once, honoring the 0.30.0 quant_method orchestration."""
        if quant_method is None:
            return self._apply_mlp(mlp_compute_input)
        return apply_moe_mlp(mlp_compute_input, quant_method)

    def _estimate_ffn_num_chunks(self, mlp_compute_input: MoEMlpComputeInput) -> int:
        if not supports_moe_ffn_chunking(mlp_compute_input):
            return 1
        num_tokens = mlp_compute_input.hidden_states.shape[0]
        return max(1, (num_tokens + self.ffn_chunk_size - 1) // self.ffn_chunk_size)

    def _apply_mlp_with_optional_chunking(
        self,
        mlp_compute_input: MoEMlpComputeInput,
        quant_method=None,
    ) -> tuple[torch.Tensor, object | None]:
        """Chunk only the local routed expert MLP, never dispatch/combine."""
        num_chunks = self._estimate_ffn_num_chunks(mlp_compute_input)
        if num_chunks <= 1:
            return self._apply_mlp_dispatched(mlp_compute_input, quant_method)

        return run_moe_ffn_in_token_chunks(
            mlp_compute_input,
            chunk_size=self.ffn_chunk_size,
            apply_mlp=lambda chunk: self._apply_mlp_dispatched(chunk, quant_method),
        )

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        quant_method=None,
    ):
        num_tokens = fused_experts_input.hidden_states.shape[0]
        num_chunks = self._e2e_chunk_count(num_tokens)
        if num_chunks == 0:
            return super().fused_experts(fused_experts_input, quant_method)
        self._log_chunk_decision(
            path="E2E_DISPATCH_FFN_COMBINE",
            raw_tokens=num_tokens,
            dispatched_tokens=None,
            num_chunks=num_chunks,
        )
        return self._fused_experts_chunked_e2e(fused_experts_input, num_chunks, quant_method)

    def _e2e_chunk_count(self, num_tokens: int) -> int:
        """Uniform per-rank chunk count for the e2e path (0 disables it).

        Only the All2AllV dispatcher is validated for repeated per-chunk
        calls; the MC2 combine kernel's sub-batch contract is unverified, so
        this override exists only on ``AlltoAllCommImpl``. Chunk boundaries
        may differ across ranks (AllToAllV handles uneven splits), but the
        chunk COUNT must be uniform across the EP group because every chunk
        issues collectives — see ``_e2e_uniform_chunk_count``.
        """
        if not self.enable_ffn_chunking:
            return 0
        return _e2e_uniform_chunk_count(num_tokens, self.ffn_chunk_size, self.moe_config)

    def _fused_experts_chunked_e2e(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        num_chunks: int,
        quant_method=None,
    ) -> FusedExpertsResult:
        """Run dispatch -> MLP -> combine per raw-token chunk.

        Each chunk is an independent dispatch/combine round trip: the
        dispatched activation (``expand_x``), the GMM intermediates and the
        combine workspace all live only for the duration of one chunk, so
        their peak scales with the per-rank chunk size instead of the full
        batch. Chunk sizes are balanced locally and may differ from other
        ranks'; only the chunk COUNT is uniform. The final outputs are
        accumulated into a single preallocated ``routed_out`` buffer, which
        ``finalize`` consumes afterwards exactly as in the unchunked path.
        """
        num_tokens = fused_experts_input.hidden_states.shape[0]
        ranges = list(balanced_chunk_ranges(num_tokens, num_chunks))

        # Earliest dispatch event lets the shared-expert stream start as soon
        # as the first chunk's dispatch begins; the latest GMM2/combine events
        # keep it from racing ahead of the last chunk (see the single-arc
        # semantics in the base implementation).
        before_dispatch_evt = torch.npu.current_stream().record_event()

        routed_out: torch.Tensor | None = None
        expert_tokens: torch.Tensor | None = None
        group_list_type = 1
        before_gmm2_evt = None
        before_combine_evt = None

        for start, end in ranges:
            chunk_input = slice_fused_experts_input_along_tokens(fused_experts_input, start, end)
            # NOTE: unlike 0.23.0, the log2phy remap already happened upstream
            # in AscendMoERunner.apply_routed_experts, so chunk_input.topk_ids
            # is already physical expert ids.
            token_dispatch_input = build_token_dispatch_input(
                fused_experts_input=chunk_input,
            )
            token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

            chunk_group_list = token_dispatch_output.group_list
            group_list_type = token_dispatch_output.group_list_type

            mlp_compute_input = build_mlp_compute_input(
                fused_experts_input=chunk_input,
                token_dispatch_output=token_dispatch_output,
                moe_config=self.moe_config,
            )
            mlp_output, before_gmm2_evt = self._apply_mlp_dispatched(mlp_compute_input, quant_method)
            before_combine_evt = torch.npu.current_stream().record_event()
            chunk_routed_out = self.token_dispatcher.token_combine(
                hidden_states=mlp_output,
                combine_metadata=token_dispatch_output.combine_metadata,
            )
            del mlp_output, token_dispatch_output

            if routed_out is None:
                routed_out = chunk_routed_out.new_empty((num_tokens, *chunk_routed_out.shape[1:]))
            routed_out[start:end].copy_(chunk_routed_out)
            del chunk_routed_out

            if chunk_group_list is not None:
                # Per-expert token counts are partial per chunk; EPLB heat
                # collection needs the total across the whole batch.
                if expert_tokens is None:
                    expert_tokens = chunk_group_list.clone()
                else:
                    expert_tokens += chunk_group_list

        assert routed_out is not None
        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=group_list_type,
            expert_tokens=expert_tokens,
        )


class FusedMC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def __init__(self, moe_config):
        super().__init__(moe_config)
        self._mega_moe_hccl_state_stale = False
        self.enable_fused_mc2 = get_ascend_config().enable_fused_mc2
        if self.enable_fused_mc2 == 1 and is_mega_moe_supported():
            self.mega_moe_symm_buffer = None
            self.get_symm_buffer_for_mega_moe, self.mega_moe = moe_utils.load_cann_mega_moe_ops()
            # Resolve the Python ABI once, before the first collective/capture.
            # The activation and its scalar parameters belong to FusedMoEConfig.
            self.mega_moe_activation_kwargs = moe_utils.select_mega_moe_activation_kwargs(
                self.mega_moe,
                activation=moe_config.activation,
                activation_clamp=moe_config.swiglu_limit if (moe_config.swiglu_limit or 0.0) > 0 else None,
                swiglu_alpha=1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha,
                swiglu_beta=0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta,
                situ_beta=moe_config.activation_situ_beta,
                situ_linear_beta=moe_config.activation_situ_linear_beta,
            )
        if self.enable_fused_mc2 == 1:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

        self.swiglu_limit = 0.0 if moe_config.swiglu_limit is None else moe_config.swiglu_limit
        self.swiglu_alpha = 1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha
        self.swiglu_beta = 0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def prepare_hccl_teardown(self) -> bool:
        """Invalidate the MegaMoe context before its MC2 group is destroyed."""
        symm_buffer = getattr(self, "mega_moe_symm_buffer", None)
        if symm_buffer is None:
            return True

        context_manager = getattr(symm_buffer, "_ctx_manager", None)
        update_group = getattr(context_manager, "update_group", None)
        if not callable(update_group):
            raise RuntimeError(
                "The installed cann_ops_transformer MegaMoe context manager "
                "does not expose update_group(); refusing to tear down its HCCL group."
            )

        self._mega_moe_hccl_state_stale = True
        logger.info("Marked MegaMoe HCCL runtime context stale before MC2 group teardown.")
        return True

    def refresh_hccl_runtime_state(self) -> bool:
        """Rebind a stale MegaMoe context to the restored MC2 communicator."""
        symm_buffer = getattr(self, "mega_moe_symm_buffer", None)
        if symm_buffer is None or not self._mega_moe_hccl_state_stale:
            return True

        device_group = get_mc2_group().device_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        group_name = backend.get_hccl_comm_name(local_rank)
        context_manager = symm_buffer._ctx_manager

        # The context tensor was created under inference mode during model
        # initialization. CANN updates it in place, so preserve that mode here.
        with torch.inference_mode():
            context_manager.update_group(group_name, symm_buffer.context)
        symm_buffer.group = device_group
        symm_buffer.rank_id = local_rank
        symm_buffer.group_name = group_name
        symm_buffer.ep_world_size = torch.distributed.get_world_size(group=device_group)
        symm_buffer.ccl_buffer_size = context_manager.ccl_buffer_size
        torch.distributed.barrier(
            group=device_group,
            device_ids=[torch.npu.current_device()],
        )
        self._mega_moe_hccl_state_stale = False
        logger.info(
            "Refreshed MegaMoe HCCL runtime context after MC2 group restore: rank=%d, world_size=%d.",
            local_rank,
            symm_buffer.ep_world_size,
        )
        return True

    def _init_mega_moe_symm_buffer(
        self,
        dispatch_quant_mode: int = 0,
        dispatch_quant_out_dtype: torch.dtype | None = None,
        *,
        is_decode_only_node: bool,
    ):
        # FusedMC2CommImpl always builds a TokenDispatcherWithMC2 (see
        # setup_moe_comm_method), which is where global_bs / ep_world_size live.
        # Assert it so mypy resolves those attributes off the base dispatcher.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        group = get_mc2_group().device_group
        # The sym buffer is allocated by get_symm_buffer_for_mega_moe, a
        # collective handshake over the EP (mc2) group. Its shape params —
        # especially num_max_tokens_per_rank — MUST be identical on every EP
        # rank, otherwise ranks allocate mismatched buffers / at different
        # times and HCCL aborts (SUSPECT REMOTE ERROR 507057). So this value
        # must be derived ONLY from rank-invariant, compile-time config,
        # NEVER from the current forward's per-rank token count.
        if self.token_dispatcher.global_bs > 0:
            # global_bs = num_tokens_per_tp_rank * ep_world_size (compile-time).
            num_max_tokens_per_rank = max(
                1,
                int(self.token_dispatcher.global_bs // self.token_dispatcher.ep_world_size),
            )
        else:
            # num_tokens_per_tp_rank, set once in TokenDispatcherWithMC2.__init__
            # from scheduler/graph config — rank-invariant.
            rank_invariant_cap = getattr(self.token_dispatcher, "max_num_tokens_per_rank", 0)
            num_max_tokens_per_rank = max(1, int(rank_invariant_cap))
        num_topk = self.moe_config.experts_per_token
        num_experts = self.moe_config.num_experts
        expert_per_rank = max(1, num_experts // int(self.token_dispatcher.ep_world_size))
        absolute_safe_max_recv_token_num = max(
            1,
            num_max_tokens_per_rank * int(self.token_dispatcher.ep_world_size) * min(num_topk, expert_per_rank),
        )

        if is_decode_only_node:
            max_recv_token_num = absolute_safe_max_recv_token_num
        else:
            # P nodes and PD-mixed nodes use the configured value. This keeps
            # the existing memory/performance tradeoff for prefill workloads.
            max_recv_token_num = get_ascend_config().mega_moe_max_tokens
            # absolute_safe_max_recv_token_num is the max value required by mega moe api
            if max_recv_token_num > absolute_safe_max_recv_token_num:
                max_recv_token_num = absolute_safe_max_recv_token_num
            logger.warning_once(
                "MegaMoe symm buffer: max_recv_token_num is set from "
                "mega_moe_max_tokens=%d (reference value) on a P or PD-mixed "
                "node. If the actual per-rank received token count after "
                "dispatch exceeds this value, precision degradation will "
                "occur. The absolute safe upper bound is %d "
                "(num_max_tokens_per_rank=%d, ep_world_size=%d, num_topk=%d, "
                "expert_per_rank=%d). Please tune mega_moe_max_tokens in "
                "additional_config based on actual expert load distribution.",
                max_recv_token_num,
                absolute_safe_max_recv_token_num,
                num_max_tokens_per_rank,
                int(self.token_dispatcher.ep_world_size),
                num_topk,
                expert_per_rank,
            )

        logger.info(
            "CANN MegaMoe sym-buffer alloc (must match across all EP ranks): ep_rank=%s ep_world=%s global_bs=%s",
            getattr(self.token_dispatcher, "ep_rank_id", "?"),
            getattr(self.token_dispatcher, "ep_world_size", "?"),
            self.token_dispatcher.global_bs,
        )

        return self.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden=self.moe_config.hidden_dim,
            intermediate_hidden=2 * self.moe_config.intermediate_size_per_partition,
            max_recv_token_num=max_recv_token_num,
            dispatch_quant_mode=dispatch_quant_mode,
            dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        )

    def _apply_cann_mega_moe(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        weights,
        is_decode_only_node: bool,
    ):
        # TokenDispatcherWithMC2 carries global_bs (used below for the mc2_mask
        # branch); assert the subtype so mypy resolves it off the base class.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        num_tokens = fused_experts_input.hidden_states.shape[0]
        num_max_tokens = self.token_dispatcher.max_num_tokens_per_rank
        if num_tokens > num_max_tokens:
            raise ValueError(
                f"MegaMoe received {num_tokens} tokens per rank, but its symmetric buffer "
                f"was allocated for at most {num_max_tokens}. Increase max_num_batched_tokens "
                "or disable fused MC2."
            )

        def to_list(x):
            return x if isinstance(x, list) else [x]

        weight1 = to_list(weights.w1)
        weight2 = to_list(weights.w2)
        # A8W4-INT MegaMoe reads N from weight1.storageShape.lastDim treated as int8 (N = lastDim*2)
        # and checks weight2.dim0 == N/2, so the weights MUST be int8-shaped (two int4 per byte), NOT
        # the eight-int4-per-int32 packing (that makes the op read N four times too small and fail
        # CheckWeight2Input). The op prototype also REQUIRES FRACTAL_NZ per expert. The W4A8 quant
        # method therefore builds per-expert int8 + FRACTAL_NZ lists (cann_mega_moe_*_weight_list) and
        # they are passed through as-is here. W8A8 weights are already int8 + FRACTAL_NZ, also as-is.
        weight_scales1 = weights.w1_scale
        weight_scales2 = weights.w2_scale
        dispatch_quant_mode, dispatch_quant_out_dtype, weight_type = moe_utils._get_cann_mega_moe_quant_settings(
            fused_experts_input.quant.quant_type
        )

        if self.mega_moe_symm_buffer is None:
            self.mega_moe_symm_buffer = self._init_mega_moe_symm_buffer(
                dispatch_quant_mode,
                dispatch_quant_out_dtype,
                is_decode_only_node=is_decode_only_node,
            )
        else:
            self.mega_moe_symm_buffer.dispatch_quant_mode = dispatch_quant_mode
            self.mega_moe_symm_buffer.dispatch_quant_out_dtype = dispatch_quant_out_dtype

        x_active_mask = None
        # Ascend 950 (A5) MegaMoe only support a null x_active_mask, and it
        # must be passed as None. But on A2/A3 it must be valid.
        if get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP):
            x_active_mask = None
        else:
            if self.token_dispatcher.global_bs == 0 and fused_experts_input.routing.mc2_mask is not None:
                # mc2_mask comes from the reserved bool buffer in
                # ascend_forward_context.set_mc2_mask. MegaMoe wants int8 as
                # the per-token active mask, so cast only when the dtype does
                # not already match — saves the kernel launch when an upstream
                # change ever flips the reserved buffer to int8.
                raw_mask = fused_experts_input.routing.mc2_mask
                if raw_mask.dtype == torch.int8:
                    x_active_mask = raw_mask.contiguous()
                else:
                    x_active_mask = raw_mask.to(torch.int8).contiguous()
        # A8W4-INT precision-compensation biases B1/B2 (l1_bias/l2_bias).
        l1_bias = weights.w1_scale_bias
        l2_bias = weights.w2_scale_bias
        # Quant methods supply the routed layer, whose activation was bound at
        # initialization. The shared communicator may belong to a later layer.
        layer = cast(torch.nn.Module, fused_experts_input.layer)
        out, expert_tokens = self.mega_moe(
            fused_experts_input.hidden_states,
            fused_experts_input.topk_ids.to(torch.int32),
            fused_experts_input.topk_weights.to(torch.float32),
            weight1,
            weight2,
            self.mega_moe_symm_buffer,
            l1_weights_sf=weight_scales1,
            l2_weights_sf=weight_scales2,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
            x_active_mask=x_active_mask,
            weight1_type=weight_type,
            weight2_type=weight_type,
            **layer.mega_moe_activation_kwargs,
        )
        # NOTE: self.expert_token_nums is only used by the
        # mega_moe path (enable_fused_mc2 == 1) as a
        # pre-allocated in/out buffer. The MegaMoe op returns a fresh
        # expert_tokens tensor that is consumed by the caller via the
        # return value, so there is nothing to keep on the instance.
        return out, expert_tokens


    def _fused_mc2_chunk_count(self, num_tokens: int) -> int:
        """Return a collective-safe chunk count for dispatch_ffn_combine."""
        if not self.enable_ffn_chunking or get_ascend_config().enable_fused_mc2 != 1:
            return 0
        return _e2e_uniform_chunk_count(num_tokens, self.ffn_chunk_size, self.moe_config)

    def _fused_mc2_chunk_max_output_size(self, num_chunks: int) -> int:
        """Bound the per-chunk routed-token workspace without extra drops.

        In uniform-token mode, each source rank contributes at most
        ``ffn_chunk_size * top_k`` routed rows. In uneven-token mode,
        ``TokenDispatcherWithMC2.global_bs`` provides the rank-invariant
        maximum source-token capacity; divide that capacity across the agreed
        chunk count instead. In the worst case all EP ranks route every row to
        experts on one destination rank, so the expression below is a strict
        upper bound for one chunk. Capping the configured capacity by that
        bound can shrink dispatch_ffn_combine's dominant workspace while never
        making overflow more likely than the unchunked configured path.

        This value depends only on rank-invariant configuration. Every rank and
        every chunk (including a shorter tail) therefore use identical
        ``max_output_size`` values for the collective fused operator.
        """
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        max_source_tokens_per_chunk = self.ffn_chunk_size
        if self.token_dispatcher.global_bs > 0:
            max_source_tokens_per_rank = (
                self.token_dispatcher.global_bs + self.token_dispatcher.ep_world_size - 1
            ) // self.token_dispatcher.ep_world_size
            max_source_tokens_per_chunk = (
                max_source_tokens_per_rank + num_chunks - 1
            ) // num_chunks
        worst_case_routed_tokens = (
            max_source_tokens_per_chunk
            * self.token_dispatcher.ep_world_size
            * self.moe_config.experts_per_token
        )
        return min(get_ascend_config().mega_moe_max_tokens, worst_case_routed_tokens)

    def _apply_dispatch_ffn_combine(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        weights: MoEWeights,
        *,
        out: torch.Tensor,
        max_output_size: int,
    ) -> torch.Tensor:
        """Apply one fused MC2 round and return this round's expert counts."""
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        assert self.expert_token_nums is not None

        # NOTE: unlike 0.23.0, the log2phy remap already happened upstream in
        # AscendMoERunner.apply_routed_experts, so topk_ids is already physical.
        torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
            x=fused_experts_input.hidden_states,
            weight1=weights.w1,
            weight2=weights.w2,
            expert_idx=fused_experts_input.topk_ids,
            scale1=weights.w1_scale,
            scale2=weights.w2_scale,
            bias1=weights.w1_scale_bias,
            bias2=weights.w2_scale_bias,
            probs=fused_experts_input.topk_weights.to(torch.float32),
            group=self.token_dispatcher.moe_all_to_all_group_name,
            max_output_size=max_output_size,
            swiglu_limit=self.swiglu_limit,
            world_size=self.token_dispatcher.ep_world_size,
            x_active_mask=fused_experts_input.routing.mc2_mask,
            out=out,
            expert_token_nums=self.expert_token_nums,
        )
        return self.expert_token_nums

    def _fused_experts_chunked(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        num_chunks: int,
        weights: MoEWeights,
    ) -> FusedExpertsResult:
        """Run dispatch_ffn_combine once per raw-token chunk."""
        assert self.expert_token_nums is not None
        num_tokens = fused_experts_input.hidden_states.shape[0]
        ranges = balanced_chunk_ranges(num_tokens, num_chunks)
        max_output_size = self._fused_mc2_chunk_max_output_size(num_chunks)

        # Use views into one full output allocation. This avoids keeping an
        # additional per-chunk output alive while preserving original token
        # order for prepare_finalize.finalize().
        routed_out = torch.empty_like(fused_experts_input.hidden_states)
        expert_tokens = torch.zeros_like(self.expert_token_nums)
        for start, end in ranges:
            chunk_input = slice_fused_experts_input_along_tokens(fused_experts_input, start, end)
            chunk_expert_tokens = self._apply_dispatch_ffn_combine(
                chunk_input,
                weights,
                out=routed_out[start:end],
                max_output_size=max_output_size,
            )
            # dispatch_ffn_combine overwrites the reusable instance buffer on
            # every call. Accumulate immediately on the same stream before the
            # next chunk reuses it.
            expert_tokens.add_(chunk_expert_tokens)

        return FusedExpertsResult(
            routed_out=routed_out,
            expert_tokens=expert_tokens,
        )

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        quant_method=None,
    ):
        if quant_method is not None:
            weights = quant_method.get_fused_mc2_weights(fused_experts_input.layer)
        else:
            # Backward-compatible fallback for legacy callers that still build
            # the weight payload through build_fused_experts_input.
            weights = fused_experts_input.weights

        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        expert_tokens = None
        if self.enable_fused_mc2 == 1:
            if _EXTRA_CTX.use_mega_moe:
                out, expert_tokens = self._apply_cann_mega_moe(
                    fused_experts_input, weights, is_decode_only_node=_EXTRA_CTX.is_decode_only_node
                )
            else:
                assert not (weights.w1_scale_bias is None or weights.w2_scale_bias is None), (
                    "w1_scale_bias and w2_scale_bias cannot be None when enable_fused_mc2=1."
                )

                raw_tokens = fused_experts_input.hidden_states.shape[0]
                num_chunks = self._fused_mc2_chunk_count(raw_tokens)
                if num_chunks > 0:
                    self._log_chunk_decision(
                        path="E2E_DISPATCH_FFN_COMBINE",
                        raw_tokens=raw_tokens,
                        dispatched_tokens=None,
                        num_chunks=num_chunks,
                    )
                    return self._fused_experts_chunked(fused_experts_input, num_chunks, weights)

                self._log_chunk_decision(
                    path="FUSED_SINGLE_PASS",
                    raw_tokens=raw_tokens,
                    dispatched_tokens=None,
                    num_chunks=1,
                    fallback_reason=(
                        "token_count_le_chunk_size"
                        if raw_tokens <= self.ffn_chunk_size
                        else "collective_chunk_count_le_one"
                    ),
                )

                out = torch.empty_like(fused_experts_input.hidden_states)
                torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                    x=fused_experts_input.hidden_states,
                    weight1=weights.w1,
                    weight2=weights.w2,
                    expert_idx=fused_experts_input.topk_ids,
                    scale1=weights.w1_scale,
                    scale2=weights.w2_scale,
                    bias1=weights.w1_scale_bias,
                    bias2=weights.w2_scale_bias,
                    probs=fused_experts_input.topk_weights.to(torch.float32),
                    group=self.token_dispatcher.moe_all_to_all_group_name,
                    max_output_size=get_ascend_config().mega_moe_max_tokens,
                    swiglu_limit=self.swiglu_limit,
                    world_size=self.token_dispatcher.ep_world_size,
                    x_active_mask=fused_experts_input.routing.mc2_mask,
                    out=out,
                    expert_token_nums=self.expert_token_nums,
                )
                expert_tokens = self.expert_token_nums
        else:
            raise ValueError(f"Wrong value of {self.enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out,
            expert_tokens=expert_tokens,
        )
