# SPDX-License-Identifier: Apache-2.0
"""CPU checks of real PCP chunk geometry and orchestration, without NPU imports.

Only platform imports, metadata kernels and attention operators are replaced.
Reference attention independently uses absolute token positions and a causal
window; it does not reconstruct the candidate's chunk algorithm.
"""

import ast
import importlib.util
import sys
from contextlib import nullcontext
from dataclasses import dataclass, make_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "vllm_ascend/attention/dsa_pcp_chunking.py"
spec = importlib.util.spec_from_file_location("pcp_chunk_geometry_under_test", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)


class Common(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "_seq_lens_cpu": None,
                "positions_cpu": None,
                "seq_lens_cpu_upper_bound": None,
                "is_prefilling": None,
                **kwargs,
            }
        )


def common_metadata(lengths, histories, request_ids=None, padding=0):
    lengths = torch.tensor(lengths, dtype=torch.int32)
    histories = torch.tensor(histories, dtype=torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    n = int(offsets[-1])
    request_ids = list(range(len(lengths))) if request_ids is None else request_ids
    positions = torch.cat([torch.arange(int(h), int(h + n)) for h, n in zip(histories, lengths)])
    slots = torch.cat([p + rid * 100000 for p, rid in zip(positions.split(lengths.tolist()), request_ids)])
    return Common(
        query_start_loc=offsets.clone(),
        query_start_loc_cpu=offsets,
        seq_lens=histories + lengths,
        seq_lens_cpu=histories + lengths,
        _seq_lens_cpu=histories + lengths,
        seq_lens_cpu_upper_bound=histories + lengths + 9,
        num_computed_tokens_cpu=histories,
        num_reqs=len(lengths),
        num_actual_tokens=n,
        num_input_tokens=n + padding,
        max_query_len=int(lengths.max()) if len(lengths) else 0,
        max_seq_len=int((histories + lengths).max()) if len(lengths) else 0,
        block_table_tensor=torch.tensor(request_ids).view(-1, 1),
        slot_mapping=torch.cat((slots, torch.full((padding,), -1))),
        positions=torch.cat((positions, torch.zeros(padding, dtype=torch.int64))),
        causal=True,
        attn_state="prefill",
        is_prefilling=lengths > 1,
    )


def chunks(common, size):
    return [
        (geometry, helper.slice_pcp_chunk_common_metadata(common, geometry))
        for geometry in helper.compute_pcp_chunk_geometries(common.query_start_loc_cpu, common.num_actual_tokens, size)
    ]


@pytest.mark.parametrize("size", [1, 4, 128, 2048])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_geometry_threshold_and_padding(size, delta):
    n = max(size + delta, 0)
    common = common_metadata([n, 0], [11, 0], padding=7)
    plan = chunks(common, size)
    assert sum(g.end - g.start for g, _ in plan) == n
    assert all(0 < g.end - g.start <= size for g, _ in plan)
    if n:
        assert torch.equal(torch.cat([c.positions for _, c in plan]), common.positions[:n])
        assert torch.equal(torch.cat([c.slot_mapping for _, c in plan]), common.slot_mapping[:n])
        assert all(c.num_input_tokens == c.num_actual_tokens for _, c in plan)


@pytest.mark.parametrize("seed", range(12))
def test_random_geometry_preserves_absolute_query_positions_and_request_identity(seed):
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, 40, (8,), generator=gen).tolist()
    histories = torch.randint(0, 200, (8,), generator=gen).tolist()
    common = common_metadata(lengths, histories, request_ids=[0, 1, 2, 0, 1, 2, 3, 3], padding=13)
    before = {name: value.clone() for name, value in vars(common).items() if isinstance(value, torch.Tensor)}
    original_rows = torch.repeat_interleave(common.block_table_tensor[:, 0], torch.tensor(lengths))
    for geometry, sliced in chunks(common, 7 + seed):
        assert torch.equal(
            torch.repeat_interleave(sliced.block_table_tensor[:, 0], sliced.query_start_loc_cpu.diff()),
            original_rows[geometry.start : geometry.end],
        )
        rebuilt_positions = torch.cat(
            [
                torch.arange(int(end - n), int(end))
                for end, n in zip(sliced.seq_lens_cpu, sliced.query_start_loc_cpu.diff())
            ]
        )
        assert torch.equal(rebuilt_positions, common.positions[geometry.start : geometry.end])
        assert torch.equal(sliced._seq_lens_cpu, sliced.seq_lens_cpu)
        assert torch.equal(sliced._num_computed_tokens_cpu, sliced.seq_lens_cpu - sliced.query_start_loc_cpu.diff())
        assert torch.equal(sliced.seq_lens_cpu_upper_bound, sliced.seq_lens_cpu + 9)
    for name, value in before.items():
        assert torch.equal(getattr(common, name), value), name


@pytest.mark.parametrize(
    "offsets,n,size", [([1, 3], 2, 1), ([0, 3, 2], 2, 1), ([0, 2], 3, 1), ([0], -1, 1), ([0], 0, 0)]
)
def test_invalid_geometry_fails_before_execution(offsets, n, size):
    with pytest.raises(ValueError):
        helper.compute_pcp_chunk_geometries(torch.tensor(offsets), n, size)


def _load_nodes(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(nodes) == len(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"),
        namespace,
    )


class GenericBase:
    def __class_getitem__(cls, _):
        return cls


@pytest.fixture
def runtime(monkeypatch):
    compressor_type = make_dataclass("AscendCompressorMetadata", ["cache", "state"], frozen=True)
    indexer_type = make_dataclass("AscendIndexerMetadata", ["compressor"], frozen=True)
    shared = dict(
        Any=object,
        torch=torch,
        dataclass=dataclass,
        replace=replace,
        nullcontext=nullcontext,
        contextmanager=__import__("contextlib").contextmanager,
        AttentionImplBase=GenericBase,
        AscendAttentionState=SimpleNamespace(ChunkedPrefill="prefill"),
        AscendCompressorMetadata=compressor_type,
        AscendIndexerMetadata=indexer_type,
        is_activation_peak_profiling=lambda: False,
        record_activation_substage=lambda *a, **k: nullcontext(),
        profile_activation=lambda *a, **k: lambda function: function,
        attention_transfer_window=nullcontext,
        wait_for_kv_layer_from_connector=MagicMock(),
        notify_kv_cache_written=MagicMock(),
        maybe_save_kv_layer_to_connector=MagicMock(),
        get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: False),
        HardwareCapability=SimpleNamespace(INPLACE_PARTIAL_ROTARY_MUL_NEGATE_SIN="negate_sin"),
        oproj_tp_enable=lambda: False,
    )
    base = dict(shared)
    _load_nodes(
        ROOT / "vllm_ascend/attention/dsa_v1.py",
        {
            "AscendDSAReqMetadata",
            "AscendDSAMetadata",
            "AscendDSALayerMetadata",
            "_require_req_metadata",
            "AscendDSAImpl",
        },
        base,
    )

    class MetadataBuilder:
        """A metadata-kernel boundary that deliberately reuses its buffers."""

        def __init__(self, *args):
            self.backing = torch.zeros(32, dtype=torch.int32)
            self.cache_group_key = "compressor_group"

        def build(self, _prefix, common, _fast, **kwargs):
            assert self.decode_threshold == 0
            assert kwargs["can_use_rope_cache"] is False
            assert kwargs["full_graph_mode"] is False
            assert kwargs["common_ratio_to_sas_metadata"] == {}
            n = common.num_reqs
            self.backing[:n].copy_(common.seq_lens)
            metadata = make_metadata(common, base)
            metadata.req_metadata.seq_lens = self.backing[:n]
            metadata.req_metadata.cache_group_key = self.cache_group_key
            return metadata

    base["AscendDSAMetadataBuilder"] = MetadataBuilder
    ns = dict(
        shared,
        dsa_v1=SimpleNamespace(**base),
        DSAPCPChunk=helper.DSAPCPChunk,
        compute_pcp_chunk_geometries=helper.compute_pcp_chunk_geometries,
        slice_pcp_chunk_common_metadata=helper.slice_pcp_chunk_common_metadata,
        logger=MagicMock(),
    )
    _load_nodes(
        ROOT / "vllm_ascend/attention/context_parallel/dsa_cp.py",
        {"AscendDSAPCPMetadata", "_DSAPCPChunkMetadataBuilder", "AscendDSAPCPMetadataBuilder", "AscendDSAPCPImpl"},
        ns,
    )
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(inplace_partial_rotary_mul=lambda *a, **k: None))
    return SimpleNamespace(**ns, base_namespace=base)


def make_metadata(common, namespace):
    req = namespace["AscendDSAReqMetadata"](
        block_table=common.block_table_tensor,
        seq_lens=common.seq_lens,
        slot_mapping=common.slot_mapping,
        storage_block_size=128,
        query_start_loc=common.query_start_loc,
        cos={"layer": torch.ones(common.num_actual_tokens, 1, 2)},
        sin={"layer": torch.zeros(common.num_actual_tokens, 1, 2)},
    )
    return namespace["AscendDSAMetadata"](
        num_actual_tokens=common.num_actual_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=common.num_reqs,
        req_metadata=req,
    )


def pcp_metadata(runtime, local, global_common, size):
    namespace = runtime.base_namespace
    global_chunks = (
        tuple(helper.DSAPCPChunk(g.start, g.end, make_metadata(c, namespace)) for g, c in chunks(global_common, size))
        if global_common.num_actual_tokens > size
        else ()
    )
    local_chunks = (
        tuple(helper.DSAPCPChunk(g.start, g.end, make_metadata(c, namespace)) for g, c in chunks(local, size))
        if local.num_actual_tokens > size
        else ()
    )
    return runtime.AscendDSAPCPMetadata.from_local_metadata(
        make_metadata(local, namespace),
        local.num_input_tokens,
        torch.arange(global_common.num_actual_tokens),
        make_metadata(global_common, namespace),
        True,
        global_chunks,
        local_chunks,
    )


@pytest.mark.parametrize("size", [1, 4, 7, 64])
def test_builder_uses_independent_buffers_and_preserves_one_token_tail(runtime, size):
    builder = runtime.AscendDSAPCPMetadataBuilder.__new__(runtime.AscendDSAPCPMetadataBuilder)
    builder._pcp_attention_chunk_size = size
    builder.kv_cache_spec = object()
    builder.layer_names = ["layer"]
    builder.vllm_config = object()
    builder.device = torch.device("cpu")
    pool = []
    common = common_metadata([size * 2 + 1], [3])
    result = builder._build_pcp_chunks(common, pool)
    assert len(result) == 3
    assert [int(c.metadata.req_metadata.seq_lens[0]) for c in result] == [size + 3, size * 2 + 3, size * 2 + 4]
    assert len({c.metadata.req_metadata.seq_lens.data_ptr() for c in result}) == 3
    assert [c.metadata.req_metadata.cache_group_key for c in result] == [
        f"compressor_group:pcp:global:{i}" for i in range(3)
    ]
    original_ids = [id(item) for item in pool]
    builder._build_pcp_chunks(common_metadata([size + 1], [101]), pool)
    assert original_ids == [id(item) for item in pool]
    assert builder._build_pcp_chunks(common_metadata([size], [101]), pool) == ()


def reference_attention(query, req, cache, window=11):
    out = []
    offsets = req.query_start_loc.tolist()
    for row, (start, end) in enumerate(zip(offsets, offsets[1:])):
        rid = int(req.block_table[row, 0])
        seq_end = int(req.seq_lens[row])
        for i in range(start, end):
            position = seq_end - (end - i)
            kv = cache[rid, max(0, position - window + 1) : position + 1]
            weights = torch.softmax(kv @ query[i], 0)
            out.append(weights @ kv)
    return torch.stack(out).unsqueeze(1) if out else query.new_empty((0, 1, query.shape[-1]))


def test_compressor_memo_separates_full_batch_global_and_local_chunks(runtime, monkeypatch):
    builder = runtime.AscendDSAPCPMetadataBuilder.__new__(runtime.AscendDSAPCPMetadataBuilder)
    builder._pcp_attention_chunk_size = 4
    builder.kv_cache_spec = builder.vllm_config = object()
    builder.layer_names = ["layer"]
    builder.device = torch.device("cpu")
    common = common_metadata([9], [3])
    global_chunks = builder._build_pcp_chunks(common, [])
    local_chunks = builder._build_pcp_chunks(common, [], cache_scope="local")
    context = SimpleNamespace(additional_kwargs={})
    ns = dict(
        torch=torch,
        get_forward_context=lambda: context,
        _COMPRESSOR_METADATA_CACHE_KEY="compressor_memo",
        get_dsa_attn_kv_plan=lambda _: SimpleNamespace(get_dsa_compressor_slot_mapping_format=lambda: 0),
    )
    _load_nodes(ROOT / "vllm_ascend/attention/dsa_v1.py", {"get_or_compute_compressor_metadata"}, ns)
    kernel = MagicMock(side_effect=lambda *args: tuple(t.clone() for t in args[2:4]))
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(compressor_metadata=kernel))
    requests = [SimpleNamespace(cache_group_key="compressor_group")]
    requests += [c.metadata.req_metadata for c in (*global_chunks, *local_chunks)]
    for i, req in enumerate(requests):
        req.full_compress_cos = req.full_compress_sin = torch.zeros(1, 2)
        req.start_pos = torch.tensor([i * 4])
        req.query_start_loc = torch.tensor([0, i + 1])
        req.block_table = torch.zeros(1, 2, dtype=torch.int32)
        req.storage_block_size = 128
        req.num_actual_reqs = 1
        req.num_compressed_tokens = i
        actual = ns["get_or_compute_compressor_metadata"](req, 4, None)
        assert torch.equal(actual[0], req.query_start_loc)
        assert torch.equal(actual[1], req.start_pos)
        # Sibling layers with the same group and chunk reuse this exact entry.
        assert ns["get_or_compute_compressor_metadata"](req, 4, None) is actual
    assert kernel.call_count == len(requests) == 7


@pytest.mark.parametrize("size", [1, 4, 8, 2048])
@pytest.mark.parametrize("rank", range(8))
def test_real_pcp_forward_matches_dense_causal_reference_and_preserves_collectives(runtime, monkeypatch, size, rank):
    # A PCP rank owns an early and a late interval of the same request.
    # Include a second request and one empty rank. Local rows are deliberately
    # discontinuous in absolute position, as in DualChunkSwap.
    lengths = [3, 5, 2] if rank < 7 else [0]
    histories = [rank * 3, 80 - rank * 5, 4] if rank < 7 else [0]
    request_ids = [0, 0, 1] if rank < 7 else [0]
    local = common_metadata(lengths, histories, request_ids, padding=3)
    global_common = common_metadata([80, 9], [5, 2], padding=7)
    meta = pcp_metadata(runtime, local, global_common, size)
    gen = torch.Generator().manual_seed(712)
    cache = torch.randn(2, 100, 2, generator=gen, dtype=torch.float64)
    global_hidden = torch.randn(global_common.num_actual_tokens, 2, generator=gen, dtype=torch.float64)
    hidden = torch.randn(local.num_input_tokens, 2, generator=gen, dtype=torch.float64)
    expected_cache = cache.clone()
    for slot, value in zip(global_common.slot_mapping, global_hidden):
        expected_cache[int(slot) // 100000, int(slot) % 100000] = value
    events = []
    impl = runtime.AscendDSAPCPImpl.__new__(runtime.AscendDSAPCPImpl)
    impl.swa_cache_layer = SimpleNamespace(prefix="swa")
    impl.compress_ratio = 1
    impl.n_local_heads = 1
    impl.head_dim = 2
    impl.nope_head_dim = 0
    impl.enable_pcp_o_proj_weight_sharding = False

    def gather(*args):
        events.append(("gather",))
        return global_hidden

    def update(layer_name, values, kv_cache, metadata):
        events.append(("cache", values.shape[0]))
        for slot, value in zip(metadata["swa"].req_metadata.slot_mapping, values):
            cache[int(slot) // 100000, int(slot) % 100000] = value

    def attention(_self, layer_name, values, kv_cache, metadata, prepared=False):
        assert prepared
        assert torch.equal(cache, expected_cache)
        events.append(("query", values.shape[0]))
        return reference_attention(values, metadata.swa.req_metadata, cache)

    def project(values, output):
        events.append(("project", values.shape[0]))
        output.copy_(values.squeeze(1))

    impl._gather_and_restore_hidden_states = gather
    impl._update_global_layer_caches = update
    impl._maybe_all_gather_pcp_o_proj_weights = lambda: events.append(("weight_gather",))
    impl._forward_o_proj = project
    monkeypatch.setattr(runtime.dsa_v1.AscendDSAImpl, "_forward_attention", attention)
    output = torch.full_like(hidden, float("nan"))
    actual = impl.forward("layer", hidden, (), {"swa": meta}, output)
    reference = reference_attention(hidden[: local.num_actual_tokens], meta.req_metadata, expected_cache)
    torch.testing.assert_close(actual[: local.num_actual_tokens], reference.squeeze(1), rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(actual[local.num_actual_tokens :]) == 0
    assert torch.equal(cache, expected_cache)
    assert events.count(("gather",)) == events.count(("weight_gather",)) == 1
    assert sum(event[0] == "project" for event in events) == int(local.num_actual_tokens > 0)
    assert max(event[1] for event in events if event[0] == "cache") <= min(size, global_common.num_actual_tokens)
    query_sizes = [event[1] for event in events if event[0] == "query"]
    assert sum(query_sizes) == local.num_actual_tokens
    assert all(value <= size for value in query_sizes)
    assert events.index(("weight_gather",)) > max(i for i, event in enumerate(events) if event[0] == "cache")


def test_local_chunks_cannot_rewrite_unprepared_cache(runtime):
    local = common_metadata([9], [0])
    meta = pcp_metadata(runtime, local, local, 4)
    impl = runtime.AscendDSAPCPImpl.__new__(runtime.AscendDSAPCPImpl)
    with pytest.raises(RuntimeError, match="prepared first"):
        impl._forward_attention("layer", torch.ones(9, 2), (), runtime.dsa_v1.AscendDSALayerMetadata(None, meta), False)


@pytest.mark.parametrize(
    "enabled,prefill,full_graph,causal,model_type,vision,multimodal,expected",
    [
        (True, True, False, True, "deepseek_v4", 0, False, True),
        (False, True, False, True, "deepseek_v4", 0, False, False),
        (True, False, False, True, "deepseek_v4", 0, False, False),
        (True, True, True, True, "deepseek_v4", 0, False, False),
        (True, True, False, False, "deepseek_v4", 0, False, False),
        (True, True, False, True, "deepseek_v41", 0, False, False),
        (True, True, False, True, "deepseek_v4", 4, False, False),
        (True, True, False, True, "deepseek_v4", 0, True, False),
    ],
)
def test_real_builder_gates_and_connects_both_chunk_plans(
    runtime, enabled, prefill, full_graph, causal, model_type, vision, multimodal, expected
):
    builder = runtime.AscendDSAPCPMetadataBuilder.__new__(runtime.AscendDSAPCPMetadataBuilder)
    builder._pcp_attention_chunk_size = 4 if enabled else 0
    builder.kv_cache_spec = object()
    builder.layer_names = ["layer"]
    builder.vllm_config = object()
    builder.device = torch.device("cpu")
    builder._pcp_global_chunk_builders = []
    builder._pcp_local_chunk_builders = []
    builder._is_decode_sharded = False
    builder.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type, vision_n_layers=vision))
    global_common = common_metadata([17], [4])
    local = common_metadata([3, 3], [4, 18], [0, 0])
    local.causal = causal
    local.mm_req_doc_ranges = {0: [(0, 2)]} if multimodal else None
    context = SimpleNamespace(
        global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([prefill]).numpy(), num_reqs=1),
        hidden_restore_idx=torch.arange(17),
    )
    builder._prepare_graph_pcp_context = lambda ctx, **k: ctx
    builder._build_global_common_attn_metadata = lambda *a: global_common
    builder._build_graph_common_attn_metadata = lambda common, *a: common
    builder._global_metadata_builder = SimpleNamespace(
        build=lambda *a, **k: make_metadata(global_common, runtime.base_namespace)
    )
    builder._build_local_common_attn_metadata = lambda *a: local
    builder._build_local_dsa_metadata = lambda *a, **k: make_metadata(local, runtime.base_namespace)
    result = builder.build(
        0,
        local,
        pcp_context=context,
        pcp_cache_group_idx=0,
        common_ratio_to_sas_metadata={},
        full_graph_mode=full_graph,
    )
    if not prefill:
        assert not isinstance(result, runtime.AscendDSAPCPMetadata)
    else:
        assert bool(result.pcp_global_chunks) == bool(result.pcp_local_chunks) == expected
        if expected:
            assert [(c.start, c.end) for c in result.pcp_global_chunks] == [(0, 4), (4, 8), (8, 12), (12, 16), (16, 17)]
            assert [(c.start, c.end) for c in result.pcp_local_chunks] == [(0, 4), (4, 6)]


@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_local_chunks_keep_compressor_and_indexer_cache_groups_aligned(runtime, monkeypatch, ratio):
    common = common_metadata([3, 6], [17, 41])
    groups = []
    for group in range(5):
        meta = pcp_metadata(runtime, common, common, 4)
        for chunk in meta.pcp_local_chunks:
            req = chunk.metadata.req_metadata
            req.block_table = torch.full_like(req.block_table, group)
        groups.append(meta)
    comp = runtime.AscendCompressorMetadata(groups[1], groups[2]) if ratio > 1 else None
    indexer = (
        runtime.AscendIndexerMetadata(runtime.AscendCompressorMetadata(groups[3], groups[4])) if ratio == 4 else None
    )
    layer = runtime.dsa_v1.AscendDSALayerMetadata(groups[1] if ratio > 1 else None, groups[0], comp, indexer)
    seen = []

    def attention(_self, name, hidden, cache, metadata, prepared):
        selected = [metadata.swa]
        if metadata.compressor is not None:
            assert metadata.attention is metadata.compressor.cache
            selected.extend((metadata.compressor.cache, metadata.compressor.state))
        if metadata.indexer is not None:
            selected.extend((metadata.indexer.compressor.cache, metadata.indexer.compressor.state))
        assert prepared
        assert [int(m.req_metadata.block_table[0, 0]) for m in selected] == list(range(len(selected)))
        assert all(
            torch.equal(m.req_metadata.query_start_loc, selected[0].req_metadata.query_start_loc) for m in selected
        )
        assert all(torch.equal(m.req_metadata.seq_lens, selected[0].req_metadata.seq_lens) for m in selected)
        seen.append(hidden.shape[0])
        return hidden.unsqueeze(1)

    monkeypatch.setattr(runtime.dsa_v1.AscendDSAImpl, "_forward_attention", attention)
    impl = runtime.AscendDSAPCPImpl.__new__(runtime.AscendDSAPCPImpl)
    hidden = torch.arange(18).reshape(9, 2).float()
    result = impl._forward_attention("layer", hidden, (), layer, True)
    assert seen == [4, 4, 1]
    assert torch.equal(result.squeeze(1), hidden)


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("size", [1, 3, 129, 2048])
def test_global_chunk_geometry_preserves_incremental_compressor_state(ratio, size):
    # Independent CPU state machine: a compressed KV is emitted when a full
    # ratio-sized window ends. Exercise partial prefix state and row changes.
    common = common_metadata([257, 129, 0, 11], [ratio - 1, 3, 0, ratio + 1])
    gen = torch.Generator().manual_seed(901)
    incoming = torch.randn(common.num_actual_tokens, 3, generator=gen, dtype=torch.float64)
    initial = {
        i: [torch.randn(3, generator=gen, dtype=torch.float64) for _ in range(int(h))]
        for i, h in enumerate(common.num_computed_tokens_cpu)
    }

    def execute(parts):
        history = {i: list(values) for i, values in initial.items()}
        compressed = {}
        for start, end, metadata in parts:
            qsl = metadata.query_start_loc_cpu.tolist()
            for row, (left, right) in enumerate(zip(qsl, qsl[1:])):
                rid = int(metadata.block_table_tensor[row, 0])
                expected_start = int(metadata.seq_lens_cpu[row]) - (right - left)
                assert len(history[rid]) == expected_start
                for value in incoming[start + left : start + right]:
                    history[rid].append(value)
                    if len(history[rid]) % ratio == 0:
                        compressed[rid, len(history[rid]) // ratio - 1] = torch.stack(history[rid][-ratio:]).sum(0)
        return history, compressed

    reference = execute([(0, common.num_actual_tokens, common)])
    candidate = execute([(g.start, g.end, c) for g, c in chunks(common, size)])
    assert candidate[1].keys() == reference[1].keys()
    for key in reference[1]:
        assert torch.equal(candidate[1][key], reference[1][key])
    for key in reference[0]:
        assert all(torch.equal(a, b) for a, b in zip(candidate[0][key], reference[0][key]))


@pytest.mark.parametrize("n", [16383, 16384, 16385, 32767])
def test_production_sized_geometry_has_complete_coverage(n):
    common = common_metadata([n // 2, n - n // 2], [71, 131])
    plan = chunks(common, 2048)
    assert torch.equal(torch.cat([c.positions for _, c in plan]), common.positions)
    assert sum(c.num_actual_tokens for _, c in plan) == n
