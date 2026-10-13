# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DSA-CP attention chunking geometry."""

import pytest
import torch

from vllm_ascend.attention.dsa_chunking import (
    compute_dsa_chunk_geometries,
    fixed_token_chunk_ranges,
)


def _reference_chunk_view(qsl, seq_lens, r0, r1):
    """Brute-force per-request intersection of the query stream with [r0, r1).

    Mirrors the conventions of _build_local_token_metadata: absent requests
    keep zero-length rows, and seq_lens is capped at the context visible at
    the range end (the causal contract of an in-forward chunk).
    """
    num_reqs = len(seq_lens)
    q_lens, seqs, starts = [], [], []
    for i in range(num_reqs):
        qs, qe = qsl[i], qsl[i + 1]
        qlen = max(0, min(qe, r1) - max(qs, r0))
        present = qlen > 0 and seq_lens[i] > 0
        if present:
            seqs.append(seq_lens[i] - max(0, qe - r1))
            starts.append(max(0, seq_lens[i] - max(0, min(qe - r0, qe - qs))))
        else:
            seqs.append(0)
            starts.append(0)
        q_lens.append(qlen)
    cu = [0]
    for qlen in q_lens:
        cu.append(cu[-1] + qlen)
    return cu, q_lens, seqs, starts


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


class TestFixedTokenChunkRanges:
    @pytest.mark.parametrize(
        "num_tokens,chunk_size,expected",
        [
            (0, 4096, []),
            (1, 4096, [(0, 1)]),
            (4096, 4096, [(0, 4096)]),
            (4097, 4096, [(0, 4096), (4096, 4097)]),
            (22000, 8192, [(0, 8192), (8192, 16384), (16384, 22000)]),
            (100, 4096, [(0, 100)]),
        ],
    )
    def test_ranges(self, num_tokens, chunk_size, expected):
        assert fixed_token_chunk_ranges(num_tokens, chunk_size) == expected

    def test_invalid(self):
        with pytest.raises(ValueError):
            fixed_token_chunk_ranges(-1, 4096)
        with pytest.raises(ValueError):
            fixed_token_chunk_ranges(10, 0)


class TestComputeDSAChunkGeometries:
    TP = 8
    CHUNK = 4096

    def _geometries(self, req_token_lens, context_before, num_actual=None, chunk_size=None):
        qsl, seq_lens = _make_batch(req_token_lens, context_before)
        num_reqs = len(req_token_lens)
        total = int(qsl[-1].item())
        num_actual = total if num_actual is None else num_actual
        num_pad = ((total + self.TP - 1) // self.TP) * self.TP
        tokens_per_rank = num_pad // self.TP
        return compute_dsa_chunk_geometries(
            qsl,
            seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual,
            local_start=0,
            local_end=tokens_per_rank,
            chunk_size=chunk_size or self.CHUNK,
        ), qsl, seq_lens, total, tokens_per_rank

    def test_single_request_spans_all_chunks(self):
        # One 12k-token request, 4k context: chunk views must partition it.
        geos, qsl, seq_lens, total, _ = self._geometries([12000], [4000])
        assert [g.c0 for g in geos] == [0, 4096, 8192]
        assert [g.c1 for g in geos] == [4096, 8192, 12000]
        for g in geos:
            ref_cu, ref_q, ref_seq, ref_start = _reference_chunk_view(
                qsl.tolist(), seq_lens.tolist(), g.c0, g.c1
            )
            assert g.qsl_k.tolist() == ref_cu
            assert g.seq_lens_k.tolist() == ref_seq
            assert g.start_pos_k.tolist() == ref_start
        # The last chunk must see the full context of the request.
        assert geos[-1].seq_lens_k.tolist() == [16000]
        # start_pos accumulates consumed tokens across chunks.
        assert [g.start_pos_k.tolist()[0] for g in geos] == [4000, 8096, 12192]
        # Chunk q_lens sum back to the full request.
        assert sum(g.qsl_k[-1].item() for g in geos) == 12000

    def test_multi_request_mixed_boundaries(self):
        # Requests straddling chunk boundaries in both directions.
        req_lens = [3000, 5000, 1500, 8000]
        ctx = [100, 0, 777, 12345]
        geos, qsl, seq_lens, _, _ = self._geometries(req_lens, ctx)
        for g in geos:
            ref_cu, ref_q, ref_seq, ref_start = _reference_chunk_view(
                qsl.tolist(), seq_lens.tolist(), g.c0, g.c1
            )
            assert g.qsl_k.tolist() == ref_cu, (g.c0, g.c1)
            assert g.seq_lens_k.tolist() == ref_seq, (g.c0, g.c1)
            assert g.start_pos_k.tolist() == ref_start, (g.c0, g.c1)
        # Global coverage: each request's tokens counted exactly once.
        counts = [0] * len(req_lens)
        for g in geos:
            lens = (g.qsl_k[1:] - g.qsl_k[:-1]).tolist()
            for i, l in enumerate(lens):
                counts[i] += l
        assert counts == req_lens

    def test_zero_length_and_future_requests_zeroed(self):
        # Request 0 fully in chunk 0; request 1 starts after all chunks end
        # only if num_actual is reduced — use long request 1 that only
        # intersects later chunks; requests absent from a chunk zero out.
        req_lens = [2000, 9000]
        ctx = [50, 60]
        geos, qsl, seq_lens, _, _ = self._geometries(req_lens, ctx)
        g0 = geos[0]  # [0, 4096): request 0 fully inside, request 1 partial
        assert (g0.qsl_k[1:] - g0.qsl_k[:-1]).tolist() == [2000, 2096]
        # Request 0's full context visible already in chunk 0.
        assert g0.seq_lens_k.tolist()[0] == 2050
        # Request 1's context capped at chunk end: 60 + 2096.
        assert g0.seq_lens_k.tolist()[1] == 2156

    def test_local_view_matches_rank_slice_intersection(self):
        # Rank slice [local_start, local_end) vs chunk ranges: local-of-chunk
        # rows must equal the intersection lengths, and l0/l1 the offsets.
        req_lens = [20000]
        ctx = [0]
        qsl, seq_lens = _make_batch(req_lens, ctx)
        total = 20000
        num_pad = ((total + self.TP - 1) // self.TP) * self.TP
        tokens_per_rank = num_pad // self.TP
        local_start = 3 * tokens_per_rank  # rank 3
        local_end = local_start + tokens_per_rank
        geos = compute_dsa_chunk_geometries(
            qsl,
            seq_lens,
            num_reqs=1,
            num_actual_tokens=total,
            local_start=local_start,
            local_end=local_end,
            chunk_size=4096,
        )
        for g in geos:
            i0 = min(max(g.c0, local_start), local_end)
            i1 = min(max(g.c1, local_start), local_end)
            expected_rows = i1 - i0
            assert g.l1 - g.l0 == expected_rows
            assert g.l0 == i0 - local_start
            assert g.l1 == i1 - local_start
            local_q_len = (g.qsl_local_k[1:] - g.qsl_local_k[:-1]).tolist()
            assert sum(local_q_len) == expected_rows
            # Local seq lens never exceed the global chunk seq lens.
            glob = g.seq_lens_k.tolist()
            loc = g.seq_lens_local_k.tolist()
            assert all(l <= max(glob[i], 0) for i, l in enumerate(loc))

    def test_conventions_hold_for_every_chunk(self):
        req_lens = [1, 8191, 4097, 3, 22000]
        ctx = [0, 10, 20, 30, 44000]
        geos, qsl, seq_lens, _, _ = self._geometries(req_lens, ctx, chunk_size=2048)
        for g in geos:
            # Monotone cu_seqlens starting at 0, non-negative lengths.
            assert g.qsl_k[0].item() == 0
            assert (g.qsl_k[1:] - g.qsl_k[:-1] >= 0).all()
            assert g.qsl_local_k[0].item() == 0
            assert (g.qsl_local_k[1:] - g.qsl_local_k[:-1] >= 0).all()
            # seq_lens >= q_len for present requests.
            q_lens = g.qsl_k[1:] - g.qsl_k[:-1]
            present = q_lens > 0
            assert (g.seq_lens_k[present] >= q_lens[present]).all()
            lq = g.qsl_local_k[1:] - g.qsl_local_k[:-1]
            lpresent = lq > 0
            assert (g.seq_lens_local_k[lpresent] >= lq[lpresent]).all()
            # maxes are >= 1 even for empty views.
            assert g.max_query_len_k >= 1 and g.max_seq_len_k >= 1
            assert g.max_local_query_len_k >= 1 and g.max_local_seq_len_k >= 1
            # compressed row budget mirrors the full-batch formula per chunk.
            chunk_tokens = g.c1 - g.c0
            assert g.num_compressed_tokens_k == min(chunk_tokens, chunk_tokens // 4 + len(req_lens))
            # qli rows derive from local seq lens.
            assert torch.equal(
                torch.div(g.seq_lens_local_k, 4, rounding_mode="floor") * 4
                + torch.remainder(g.seq_lens_local_k, 4),
                g.seq_lens_local_k,
            )

    @pytest.mark.parametrize("N,chunk", [(1, 4096), (100, 4096), (4097, 4096), (22000, 8192), (22000, 1024)])
    def test_partition_exact(self, N, chunk):
        # A single request: chunk cu_seqlens concatenate to the full stream.
        qsl, seq_lens = _make_batch([N], [321])
        geos = compute_dsa_chunk_geometries(
            qsl,
            seq_lens,
            num_reqs=1,
            num_actual_tokens=N,
            local_start=0,
            local_end=N,
            chunk_size=chunk,
        )
        assert sum(g.qsl_k[-1].item() for g in geos) == N
        assert [g.c0 for g in geos] == [r[0] for r in fixed_token_chunk_ranges(N, chunk)]

    def test_invalid_inputs(self):
        qsl, seq_lens = _make_batch([10], [5])
        with pytest.raises(ValueError):
            compute_dsa_chunk_geometries(
                qsl, seq_lens, num_reqs=1, num_actual_tokens=10, local_start=0, local_end=10,
                chunk_size=4, compressor_ratio=128,
            )
        with pytest.raises(ValueError):
            compute_dsa_chunk_geometries(
                qsl, seq_lens, num_reqs=0, num_actual_tokens=10, local_start=0, local_end=10,
                chunk_size=4,
            )
        # qsl must end at num_actual_tokens.
        with pytest.raises(ValueError):
            compute_dsa_chunk_geometries(
                qsl, seq_lens, num_reqs=1, num_actual_tokens=9, local_start=0, local_end=10,
                chunk_size=4,
            )
