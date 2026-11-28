# PCP restore scatter

`checked_pcp_scatter` restores byte ranges from a completed PCP AllGather on A3
and A5. The production caller is `PcpLayerwiseLoad`. Each rank contributes its
aligned KV payload followed by a 512-byte trailer; the first trailer byte is
zero for a successful pool read and nonzero for a failed read.

## Interface and effects

- `pointers`: contiguous int64 tensor `(3, N)` containing source addresses,
  destination addresses and byte lengths. Source ranges must stay within rank
  payloads and exclude trailers. Destination ranges must not overlap.
- `gathered`: contiguous uint8 storage containing all rank packets.
- `result`: one uint8 element, written to zero on success or one on failure.
- `payload_bytes`: aligned payload length per rank, excluding the trailer.
- `world_size`, `status_bytes`, `block_size`: topology, trailer and copy tile.

Every program checks every rank's status before any destination write. There is
no inter-program publication race: a failed peer suppresses all writes. Program
zero always writes the result, even when there are no descriptors. On success,
copying is byte-exact, including packed quantized KV. No numeric conversion is
performed. AllGather completion and pointer/storage lifetimes are the caller's
responsibility. The caller copies the result to pinned host storage and checks
it only after its existing stream-completion wait.

## Execution and specialization

One program copies one bounded range. A3 uses 4 KiB tiles and A5 uses 64 KiB
tiles, selected by the existing hardware profile. World size, trailer size and
tile size are compile-time constants. Descriptor count and payload size are
runtime arguments marked `do_not_specialize`; changing prefix length does not
intentionally create a new specialization. This eager-only restore path does
not register a graph warmup owner. Actual compiler variants and performance
must be inspected during later NPU validation.

## Validation

CPU tests cover descriptor mapping across rank trailers, failure propagation,
padding, empty rank shards, growth/reuse, and deferred stream dependencies.
CPU/Gloo tests exercise actual multi-process collectives; the NPU scatter is
replaced by a CPU byte-copy reference there. The NPU test is
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_pcp_restore.py`.
Its byte-exact comparisons, error cases and per-family compilation remain
pending until NPU testing is authorized; CPU success does not establish them.
