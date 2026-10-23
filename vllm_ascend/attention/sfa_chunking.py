# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-chunking plan for the SFA sparse attention prefill stage.

The SFA forward (``AscendSFAImpl.forward``) processes a whole scheduler step
as one batch: the fused q/kv preprocess (which also writes the paged SFA KV
cache), the lightning indexer (k path, indexer cache write, top-k selection)
and the sparse flash attention op all run on the full token batch, so their
transient activations scale with ``max_num_batched_tokens``. When
``enable_attention_chunking`` is set, the forward instead walks the token
stream in fixed chunks. Every chunk re-runs the full stage list on its token
slice, which keeps the per-chunk causality invariant identical to
scheduler-level chunked prefill: a chunk's queries only attend to keys at or
before the chunk end, and every chunk writes its own KV and indexer cache
rows before selecting and attending.

SFA shards heads, not tokens: every rank holds the whole (unpadded) token
stream, so the context-parallel rank-slice dimension of
``dsa_chunking.compute_dsa_chunk_geometries`` degenerates — the local range of
a chunk equals the chunk itself. The pure geometry is therefore reused with
``local_start=0, local_end=num_actual_tokens``; this module layers the
per-chunk metadata views on top:

- ``build_sfa_attn_metadata_chunk_views`` derives one ``AscendSFAMetadata``
  view per chunk (sliced slot mapping / rope / positions, chunk-rebased
  cumulative query lens and capped seq lens) consumed by the fused preprocess
  and the sparse attention op;
- ``build_sfa_indexer_metadata_chunk_views`` derives the matching
  ``AscendSFAIndexerMetadata`` views for the indexer's cache write and top-k
  selection.

All geometry math stays on small CPU tensors (see ``dsa_chunking``); the views
transfer only the per-chunk int tensors to the device once per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from vllm_ascend.attention.dsa_chunking import (
    DSAChunkGeometry,
    compute_dsa_chunk_geometries,
)


@dataclass
class SFAChunkMeta:
    """One chunk's geometry plus its per-chunk metadata views."""

    geometry: DSAChunkGeometry
    # Chunk view of the step's AscendSFAMetadata (fused preprocess + sparse
    # attention inputs). ``cum_query_lens``/``seq_lens`` are the chunk-rebased
    # device tensors from the geometry.
    attn_metadata: object = None
    # Chunk view of the owning indexer group's AscendSFAIndexerMetadata
    # (indexer cache write + top-k selection), None for indexer-less layers.
    indexer_metadata: object = None


@dataclass
class SFAChunkPlan:
    """Chunking plan attached to ``AscendSFAMetadata.chunks``."""

    chunk_size: int
    num_reqs: int
    chunks: list[SFAChunkMeta] = field(default_factory=list)


def compute_sfa_chunk_geometries(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    *,
    num_reqs: int,
    num_actual_tokens: int,
    chunk_size: int,
) -> list[DSAChunkGeometry]:
    """Per-chunk geometries for the non-CP SFA layout.

    SFA shards attention heads across ranks and keeps the token stream
    replicated, so this rank's token slice is the whole unpadded stream and
    the CP-local range of every chunk equals the chunk itself.
    """
    return compute_dsa_chunk_geometries(
        query_start_loc_cpu,
        seq_lens_cpu,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        local_start=0,
        local_end=num_actual_tokens,
        chunk_size=chunk_size,
    )


def sfa_attention_chunking_eligible(
    *,
    enable_attention_chunking: bool,
    attention_chunk_size: int,
    num_prefills: int,
    num_decode_tokens: int,
    num_input_tokens: int,
    num_actual_tokens: int,
) -> bool:
    """Whether an SFA prefill step should run the chunked forward.

    Only pure prefill steps are chunked: mixed decode+prefill steps and
    decode-only steps keep the unchunked path, so the per-chunk metadata
    views never have to model a decode/prefill split. Chunk views address
    the unpadded stream row by row, so padded (graph/DP) steps also fall
    back — chunking already requires eager mode, which keeps the two
    counts equal in practice.
    """
    if not enable_attention_chunking:
        return False
    if num_prefills <= 0 or num_decode_tokens > 0:
        return False
    if num_input_tokens != num_actual_tokens:
        return False
    # A single chunk already covering the step gains nothing.
    return num_actual_tokens > attention_chunk_size


def _rebased_device_tensors(
    geometry: DSAChunkGeometry,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-rebased cu-query-lens / capped seq-lens on the step's device.

    ``qsl_k[1:]`` follows the ``cum_query_lens`` convention (per-request
    cumulative ends); ``seq_lens_k`` is the context visible at the chunk end,
    the causal contract of an in-forward chunk.
    """
    qsl_k = geometry.qsl_k.to(device=cum_query_lens.device, dtype=cum_query_lens.dtype)
    seq_lens_k = geometry.seq_lens_k.to(device=seq_lens.device, dtype=seq_lens.dtype)
    return qsl_k, seq_lens_k


def build_sfa_attn_metadata_chunk_views(
    attn_metadata,
    geometries: list[DSAChunkGeometry],
) -> list:
    """One ``AscendSFAMetadata`` view per chunk via ``dataclasses.replace``.

    The views share the step's block table and cache tensors and override
    only the per-token rows (slot mapping, rope, positions) and the
    chunk-rebased sequence lengths. Zero-length request rows are already the
    normal input shape of the downstream kernels, matching the DSA-CP chunk
    conventions.
    """
    views = []
    for geometry in geometries:
        c0, c1 = geometry.c0, geometry.c1
        qsl_k, seq_lens_k = _rebased_device_tensors(
            geometry, attn_metadata.cum_query_lens, attn_metadata.seq_lens
        )
        cos = None if attn_metadata.cos is None else attn_metadata.cos[c0:c1]
        sin = None if attn_metadata.sin is None else attn_metadata.sin[c0:c1]
        views.append(
            replace(
                attn_metadata,
                num_input_tokens=c1 - c0,
                num_actual_tokens=c1 - c0,
                slot_mapping=attn_metadata.slot_mapping[c0:c1],
                pcp_slot_mapping=None,
                cos=cos,
                sin=sin,
                positions=None if attn_metadata.positions is None else attn_metadata.positions[c0:c1],
                cum_query_lens=qsl_k[1:],
                seq_lens=seq_lens_k,
                query_start_loc=qsl_k,
                max_query_len=geometry.max_query_len_k,
                max_seq_len=geometry.max_seq_len_k,
                num_decode_tokens=0,
            )
        )
    return views


def build_sfa_indexer_metadata_chunk_views(
    indexer_metadata,
    geometries: list[DSAChunkGeometry],
) -> list:
    """One ``AscendSFAIndexerMetadata`` view per chunk.

    The indexer's cache write consumes the sliced slot mapping and rope; the
    top-k selection consumes the chunk-rebased query/key lengths. Grouping
    fields (``group_len`` etc.) describe the cache layout, not the token
    batch, and are shared with the step.
    """
    views = []
    for geometry in geometries:
        c0, c1 = geometry.c0, geometry.c1
        qsl_k, seq_lens_k = _rebased_device_tensors(
            geometry, indexer_metadata.cum_query_lens, indexer_metadata.seq_lens
        )
        views.append(
            replace(
                indexer_metadata,
                num_actual_tokens=c1 - c0,
                slot_mapping=indexer_metadata.slot_mapping[c0:c1],
                cos=None if indexer_metadata.cos is None else indexer_metadata.cos[c0:c1],
                sin=None if indexer_metadata.sin is None else indexer_metadata.sin[c0:c1],
                cum_query_lens=qsl_k[1:],
                seq_lens=seq_lens_k,
                actual_seq_lengths_query=qsl_k[1:],
                actual_seq_lengths_key=seq_lens_k,
                num_decode_tokens=0,
            )
        )
    return views
