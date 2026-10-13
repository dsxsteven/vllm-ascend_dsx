# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-chunking plan for the DSA-CP sparse attention prefill stage.

The DSA-CP forward (``AscendDSACPImpl._forward``) processes a whole scheduler
step as one batch: q prolog, SWA KV write, indexer cache update, top-k select,
compressor and the sparse attention op all run on the full token batch, so
their transient activations scale with ``max_num_batched_tokens``. When
``enable_attention_chunking`` is set, the forward instead walks the global
token stream in fixed chunks. Every chunk re-runs the full stage list on its
token slice, which keeps the per-chunk causality invariant identical to
scheduler-level chunked prefill: a chunk's queries only attend to keys at or
before the chunk end, and every chunk writes its own KV/indexer/compressor
cache rows before selecting and attending.

This module owns the pure geometry: how the flattened per-request query
intervals intersect a contiguous chunk range, and how they further intersect
each rank's CP-local token slice. All math runs on small CPU tensors so it is
unit-testable without a device; the metadata builder transfers the results and
attaches the per-chunk SAS/QLI metadata (see ``DSAChunkPlan``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Mirrors dsa_v1._num_compressor_metadata_rows for one chunk: at most one
# rounding row per request on top of the compressed token count.
_COMPRESSOR_ROUNDING_PER_REQ = 1


@dataclass(frozen=True)
class DSAChunkGeometry:
    """CPU-only view of one chunk's token geometry (per step, group-agnostic)."""

    index: int
    # Global token range of the chunk within the actual (unpadded) stream.
    c0: int
    c1: int
    # This rank's local rows of the chunk, as offsets into the rank's local
    # token slice ([0, tokens_per_rank)). Empty when the chunk lives entirely
    # on other ranks; global-side work still runs for such chunks.
    l0: int
    l1: int
    # Global chunk view: rebased query cu_seqlens (zero-length rows for
    # requests absent from the chunk) and the KV length visible at the chunk
    # end for each request.
    qsl_k: torch.Tensor  # [num_reqs + 1] int32
    seq_lens_k: torch.Tensor  # [num_reqs] int32
    # Compressed-position start for the chunk per request: original start_pos
    # plus the request tokens already consumed by earlier chunks.
    start_pos_k: torch.Tensor  # [num_reqs] int32
    # Local-of-chunk view (chunk ∩ rank slice), same conventions as
    # ``_build_local_token_metadata`` outputs.
    qsl_local_k: torch.Tensor  # [num_reqs + 1] int32
    seq_lens_local_k: torch.Tensor  # [num_reqs] int32
    # Upper bounds the metadata operators want as python ints (>= 1).
    max_query_len_k: int
    max_seq_len_k: int
    max_local_query_len_k: int
    max_local_seq_len_k: int
    # Row budget for the chunk's compressor metadata outputs.
    num_compressed_tokens_k: int


@dataclass
class DSAChunkMeta:
    """One chunk's geometry plus device tensors and per-group metadata."""

    geometry: DSAChunkGeometry
    # Device copies of the geometry tensors (transferred once per step).
    qsl_k: torch.Tensor = None
    seq_lens_k: torch.Tensor = None
    start_pos_k: torch.Tensor = None
    qsl_local_k: torch.Tensor = None
    seq_lens_local_k: torch.Tensor = None
    # Per-chunk QLI rows derived from seq_lens_local_k.
    qli_seqused_k: torch.Tensor = None
    qli_cmp_residual_k: torch.Tensor = None
    # Opaque metadata buffers produced by the SAS/QLI metadata operators.
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None


@dataclass
class DSAChunkPlan:
    """Chunking plan attached to ``AscendDSAReqMetadata.chunks``."""

    chunk_size: int
    num_reqs: int
    chunks: list[DSAChunkMeta] = field(default_factory=list)


def fixed_token_chunk_ranges(num_tokens: int, chunk_size: int) -> list[tuple[int, int]]:
    """Ordered ``[start, end)`` ranges of at most ``chunk_size`` tokens each."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [(start, min(start + chunk_size, num_tokens)) for start in range(0, num_tokens, chunk_size)]


def _intersect_token_range(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    range_start: int,
    range_end: int,
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect per-request query intervals with the global range.

    Mirrors the clamp/cumsum math of ``AscendDSACPMetadataBuilder
    ._build_local_token_metadata`` so the outputs follow the same conventions:
    requests absent from the range keep zero-length rows, and the returned
    seq_lens is capped at the context visible at the range end (tokens of the
    same request beyond the range end are excluded, which is exactly the
    causal contract of an in-forward chunk).
    """
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    sl = seq_lens[:num_reqs].to(torch.int32)
    query_start = qsl[:-1].clamp(min=range_start, max=range_end)
    query_end = qsl[1:].clamp(min=range_start, max=range_end)
    query_lens = query_end - query_start
    # Tokens of each request that live after the range end.
    tokens_after = qsl[1:] - query_end
    valid = (query_lens > 0) & (sl > 0)
    seq_lens_k = torch.where(valid, (sl - tokens_after).clamp_min(0), torch.zeros_like(sl))
    qsl_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(query_lens, dim=0).to(torch.int32)]
    )
    return qsl_k, query_lens, seq_lens_k


def _chunk_start_pos(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    chunk_start: int,
    num_reqs: int,
    query_lens_in_chunk: torch.Tensor,
) -> torch.Tensor:
    """Per-request compressed-position start for one chunk.

    ``start_pos_k = seq_lens - tokens_from_chunk_start_onward``: the original
    step start_pos plus the request tokens consumed by earlier chunks.
    """
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    sl = seq_lens[:num_reqs].to(torch.int32)
    # Request tokens at or after the chunk start: for requests fully inside
    # or after the chunk this is the full query length (clamp from above),
    # for straddlers only the post-boundary part.
    query_lens_full = qsl[1:] - qsl[:-1]
    tokens_from_chunk_start = (qsl[1:] - chunk_start).clamp(min=0).minimum(query_lens_full)
    in_chunk = (query_lens_in_chunk > 0) & (sl > 0)
    return torch.where(
        in_chunk,
        (sl - tokens_from_chunk_start).clamp_min(0),
        torch.zeros_like(sl),
    )


def compute_dsa_chunk_geometries(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    *,
    num_reqs: int,
    num_actual_tokens: int,
    local_start: int,
    local_end: int,
    chunk_size: int,
    compressor_ratio: int = 4,
) -> list[DSAChunkGeometry]:
    """Build the per-chunk geometries for one scheduler step.

    All inputs are CPU tensors (``query_start_loc_cpu`` ends at
    ``num_actual_tokens``). ``local_start``/``local_end`` delimit this rank's
    CP token slice in the padded global stream; the intersection with each
    chunk yields the rank-relative ``[l0, l1)`` rows used for local-side work
    (q prolog, top-k, attention output accumulation).
    """
    if compressor_ratio != 4:
        raise ValueError(f"DSA attention chunking supports compressor_ratio 4, got {compressor_ratio}")
    if num_reqs <= 0:
        raise ValueError(f"num_reqs must be positive, got {num_reqs}")
    if num_actual_tokens > 0 and query_start_loc_cpu[num_reqs].item() != num_actual_tokens:
        raise ValueError(
            "query_start_loc must end at num_actual_tokens "
            f"({query_start_loc_cpu[num_reqs].item()} != {num_actual_tokens})"
        )

    geometries: list[DSAChunkGeometry] = []
    for index, (c0, c1) in enumerate(fixed_token_chunk_ranges(num_actual_tokens, chunk_size)):
        qsl_k, query_lens_k, seq_lens_k = _intersect_token_range(
            query_start_loc_cpu, seq_lens_cpu, c0, c1, num_reqs
        )
        start_pos_k = _chunk_start_pos(query_start_loc_cpu, seq_lens_cpu, c0, num_reqs, query_lens_k)
        # Clamp both endpoints into the rank slice before subtracting so
        # chunks entirely outside the slice yield an empty (equal) local
        # range instead of negative offsets.
        l0 = min(max(c0, local_start), local_end)
        l1 = min(max(c1, local_start), local_end)
        qsl_local_k, local_query_lens_k, seq_lens_local_k = _intersect_token_range(
            query_start_loc_cpu, seq_lens_cpu, l0, l1, num_reqs
        )
        chunk_tokens = c1 - c0
        geometries.append(
            DSAChunkGeometry(
                index=index,
                c0=c0,
                c1=c1,
                l0=l0 - local_start,
                l1=l1 - local_start,
                qsl_k=qsl_k,
                seq_lens_k=seq_lens_k,
                start_pos_k=start_pos_k,
                qsl_local_k=qsl_local_k,
                seq_lens_local_k=seq_lens_local_k,
                max_query_len_k=max(1, int(query_lens_k.max().item())),
                max_seq_len_k=max(1, int(seq_lens_k.max().item())),
                max_local_query_len_k=max(1, int(local_query_lens_k.max().item())),
                max_local_seq_len_k=max(1, int(seq_lens_local_k.max().item())),
                num_compressed_tokens_k=min(
                    chunk_tokens,
                    chunk_tokens // compressor_ratio + num_reqs * _COMPRESSOR_ROUNDING_PER_REQ,
                ),
            )
        )
    return geometries
