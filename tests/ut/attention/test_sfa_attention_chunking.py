# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the SFA attention chunking plan and metadata views."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.dsa_chunking import compute_dsa_chunk_geometries
from vllm_ascend.attention.sfa_chunking import (
    build_sfa_attn_metadata_chunk_views,
    build_sfa_indexer_metadata_chunk_views,
    compute_sfa_chunk_geometries,
    sfa_attention_chunking_eligible,
)


def _make_batch(req_token_lens, context_before):
    """qsl/seq_lens CPU tensors from per-request query and context lengths."""
    qsl = [0]
    for qlen in req_token_lens:
        qsl.append(qsl[-1] + qlen)
    seq_lens = [q + c for q, c in zip(req_token_lens, context_before)]
    return (
        torch.tensor(qsl, dtype=torch.int32),
        torch.tensor(seq_lens, dtype=torch.int32),
    )


def _make_attn_metadata(qsl, seq_lens, num_input_tokens=None):
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

    num_reqs = seq_lens.shape[0]
    total = int(qsl[-1])
    if num_input_tokens is None:
        num_input_tokens = total
    return AscendSFAMetadata(
        num_actual_tokens=total,
        num_input_tokens=num_input_tokens,
        slot_mapping=torch.arange(num_input_tokens, dtype=torch.int32) * 3,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        cum_query_lens=qsl[1 : num_reqs + 1].to(torch.int64),
        block_table=torch.zeros(num_reqs, 4, dtype=torch.int32),
        sin=torch.arange(num_input_tokens, dtype=torch.float32).view(-1, 1),
        cos=torch.arange(num_input_tokens, dtype=torch.float32).view(-1, 1) * 2,
        positions=torch.arange(num_input_tokens, dtype=torch.int64),
        query_start_loc=qsl,
        num_prefills=1,
        num_decode_tokens=0,
    )


def _make_indexer_metadata(qsl, seq_lens):
    from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata

    num_reqs = seq_lens.shape[0]
    total = int(qsl[-1])
    return AscendSFAIndexerMetadata(
        num_actual_tokens=total,
        slot_mapping=torch.arange(total, dtype=torch.int32) * 5,
        seq_lens=seq_lens,
        cum_query_lens=qsl[1 : num_reqs + 1].to(torch.int64),
        block_table=torch.zeros(num_reqs, 4, dtype=torch.int32),
        sin=torch.arange(total, dtype=torch.float32).view(-1, 1),
        cos=torch.arange(total, dtype=torch.float32).view(-1, 1),
        actual_seq_lengths_query=qsl[1 : num_reqs + 1].to(torch.int64),
        actual_seq_lengths_key=seq_lens.clone(),
    )


class TestSfaChunkingEligibility:
    def test_disabled(self):
        assert not sfa_attention_chunking_eligible(
            enable_attention_chunking=False,
            attention_chunk_size=8192,
            num_prefills=1,
            num_decode_tokens=0,
            num_input_tokens=20000,
            num_actual_tokens=20000,
        )

    def test_decode_only_or_mixed_steps_fall_back(self):
        for num_prefills, num_decode_tokens in ((0, 0), (0, 100), (1, 100)):
            assert not sfa_attention_chunking_eligible(
                enable_attention_chunking=True,
                attention_chunk_size=8192,
                num_prefills=num_prefills,
                num_decode_tokens=num_decode_tokens,
                num_input_tokens=20000,
                num_actual_tokens=20000,
            )

    def test_padded_step_falls_back(self):
        assert not sfa_attention_chunking_eligible(
            enable_attention_chunking=True,
            attention_chunk_size=8192,
            num_prefills=1,
            num_decode_tokens=0,
            num_input_tokens=16384,
            num_actual_tokens=16000,
        )

    @pytest.mark.parametrize(
        "num_tokens,chunk_size,expected",
        [
            (8192, 8192, False),  # a single chunk gains nothing
            (8193, 8192, True),
            (20000, 8192, True),
        ],
    )
    def test_size_threshold(self, num_tokens, chunk_size, expected):
        assert (
            sfa_attention_chunking_eligible(
                enable_attention_chunking=True,
                attention_chunk_size=chunk_size,
                num_prefills=1,
                num_decode_tokens=0,
                num_input_tokens=num_tokens,
                num_actual_tokens=num_tokens,
            )
            is expected
        )


class TestSfaChunkGeometries:
    def test_local_range_equals_global(self):
        """Non-CP SFA: every rank holds the whole stream, so the CP-local
        slice of each chunk is the chunk itself."""
        qsl, seq_lens = _make_batch([12000, 8000], [5000, 200])
        geometries = compute_sfa_chunk_geometries(
            qsl, seq_lens, num_reqs=2, num_actual_tokens=20000, chunk_size=8192
        )
        assert [(g.c0, g.c1) for g in geometries] == [(0, 8192), (8192, 16384), (16384, 20000)]
        for g in geometries:
            assert (g.l0, g.l1) == (g.c0, g.c1)
            assert torch.equal(g.qsl_local_k, g.qsl_k)
            assert torch.equal(g.seq_lens_local_k, g.seq_lens_k)

    def test_matches_dsa_geometry_with_full_local_slice(self):
        """compute_sfa_chunk_geometries is compute_dsa_chunk_geometries with
        local_start=0, local_end=num_actual_tokens."""
        qsl, seq_lens = _make_batch([9000, 3000, 1], [0, 15000, 999999])
        kwargs = dict(num_reqs=3, num_actual_tokens=12001, chunk_size=4096)
        sfa = compute_sfa_chunk_geometries(qsl, seq_lens, **kwargs)
        dsa = compute_dsa_chunk_geometries(qsl, seq_lens, local_start=0, local_end=12001, **kwargs)
        assert len(sfa) == len(dsa)
        for a, b in zip(sfa, dsa):
            assert (a.c0, a.c1, a.l0, a.l1) == (b.c0, b.c1, b.l0, b.l1)
            for field in ("qsl_k", "seq_lens_k", "start_pos_k", "qsl_local_k", "seq_lens_local_k"):
                assert torch.equal(getattr(a, field), getattr(b, field))


class TestSfaAttnMetadataChunkViews:
    def test_views_slice_and_rebase(self):
        qsl, seq_lens = _make_batch([12000, 8000], [5000, 200])
        metadata = _make_attn_metadata(qsl, seq_lens)
        geometries = compute_sfa_chunk_geometries(
            qsl, seq_lens, num_reqs=2, num_actual_tokens=20000, chunk_size=8192
        )
        views = build_sfa_attn_metadata_chunk_views(metadata, geometries)
        assert len(views) == 3
        total = 0
        for geometry, view in zip(geometries, views, strict=True):
            c0, c1 = geometry.c0, geometry.c1
            assert view.num_input_tokens == view.num_actual_tokens == c1 - c0
            total += c1 - c0
            # per-token rows are plain slices of the step tensors
            assert torch.equal(view.slot_mapping, metadata.slot_mapping[c0:c1])
            assert torch.equal(view.cos, metadata.cos[c0:c1])
            assert torch.equal(view.sin, metadata.sin[c0:c1])
            assert torch.equal(view.positions, metadata.positions[c0:c1])
            # sequence lengths follow the chunk conventions: cumulative query
            # ends rebased from zero, seq lens capped at the chunk end
            assert torch.equal(view.cum_query_lens, geometry.qsl_k[1:].to(torch.int64))
            assert torch.equal(view.seq_lens, geometry.seq_lens_k)
            assert view.num_decode_tokens == 0
        assert total == 20000

    def test_chunk_seq_lens_causal_contract(self):
        """A request straddling a chunk boundary: the earlier chunk's seq_len
        is capped at the tokens visible at the chunk end."""
        qsl, seq_lens = _make_batch([10000], [3000])  # one 10k prefill, 3k context
        metadata = _make_attn_metadata(qsl, seq_lens)
        geometries = compute_sfa_chunk_geometries(
            qsl, seq_lens, num_reqs=1, num_actual_tokens=10000, chunk_size=4096
        )
        views = build_sfa_attn_metadata_chunk_views(metadata, geometries)
        # chunk 0: 4096 new tokens + 3000 context; chunk 2: the last 1816
        assert views[0].seq_lens.tolist() == [7096]
        assert views[1].seq_lens.tolist() == [3000 + 8192]
        assert views[2].seq_lens.tolist() == [13000]
        assert [v.cum_query_lens.tolist()[0] for v in views] == [4096, 4096, 1808]

    def test_shared_fields_unchanged(self):
        qsl, seq_lens = _make_batch([9000], [0])
        metadata = _make_attn_metadata(qsl, seq_lens)
        geometries = compute_sfa_chunk_geometries(
            qsl, seq_lens, num_reqs=1, num_actual_tokens=9000, chunk_size=4096
        )
        views = build_sfa_attn_metadata_chunk_views(metadata, geometries)
        for view in views:
            assert view.block_table is metadata.block_table
            assert view.attn_state is metadata.attn_state


class TestSfaIndexerMetadataChunkViews:
    def test_views_slice_and_rebase(self):
        qsl, seq_lens = _make_batch([12000, 8000], [5000, 200])
        metadata = _make_indexer_metadata(qsl, seq_lens)
        geometries = compute_sfa_chunk_geometries(
            qsl, seq_lens, num_reqs=2, num_actual_tokens=20000, chunk_size=8192
        )
        views = build_sfa_indexer_metadata_chunk_views(metadata, geometries)
        for geometry, view in zip(geometries, views, strict=True):
            c0, c1 = geometry.c0, geometry.c1
            assert view.num_actual_tokens == c1 - c0
            assert torch.equal(view.slot_mapping, metadata.slot_mapping[c0:c1])
            assert torch.equal(view.cos, metadata.cos[c0:c1])
            assert torch.equal(view.sin, metadata.sin[c0:c1])
            assert torch.equal(view.actual_seq_lengths_query, geometry.qsl_k[1:].to(torch.int64))
            assert torch.equal(view.actual_seq_lengths_key, geometry.seq_lens_k)
            # grouping fields describe the cache layout, not the batch
            assert view.block_table is metadata.block_table


class TestTopkIndexCacheOffsets:
    def test_update_then_read_with_offset(self):
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

        buffer = torch.zeros(20, 7, dtype=torch.int32)
        impl = SimpleNamespace(topk_indices_buffer=buffer)
        chunk_a = torch.full((12, 7), 11, dtype=torch.int32)
        chunk_b = torch.full((8, 7), 22, dtype=torch.int32)
        AscendSFAImpl._update_indexcache_topk_indices(impl, chunk_a, offset=0)
        AscendSFAImpl._update_indexcache_topk_indices(impl, chunk_b, offset=12)
        got_a = AscendSFAImpl._get_indexcache_topk_indices(impl, 12, offset=0)
        got_b = AscendSFAImpl._get_indexcache_topk_indices(impl, 8, offset=12)
        assert got_a.shape == (12, 1, 7)  # 2D cache rows are read as (n, 1, k)
        assert (got_a == 11).all()
        assert (got_b == 22).all()
        # 3D inputs are squeezed onto the 2D cache
        AscendSFAImpl._update_indexcache_topk_indices(impl, chunk_b.unsqueeze(1), offset=0)
        assert (AscendSFAImpl._get_indexcache_topk_indices(impl, 8, offset=0).squeeze(1) == 22).all()

    def test_default_offsets_preserve_unchunked_behavior(self):
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

        buffer = torch.zeros(10, 4, dtype=torch.int32)
        impl = SimpleNamespace(topk_indices_buffer=buffer)
        AscendSFAImpl._update_indexcache_topk_indices(impl, torch.full((10, 4), 9, dtype=torch.int32))
        assert (AscendSFAImpl._get_indexcache_topk_indices(impl, 10) == 9).all()
