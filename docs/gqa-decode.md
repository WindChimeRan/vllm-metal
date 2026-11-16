# GQA decode routing

The default paged attention path runs ordinary single- and multi-request
decode on a GQA producer for the geometries below. The producer splits each
request's KV into 256-token partitions. It has FP16/BF16 specializations for
head dimensions 128/256 with kernel block16, plus head256 with kernel block32:
6 active shader specializations.

The partition size is fixed. A query head walks its partition serially, so
partitions are the producer's only parallelism over context length, and a
finer partition fills the GPU sooner. Routing decides only whether to use the
producer, from a fixed work budget over complete partitions; the detected core
count is its only device input. This is a backend-internal split-KV choice
like [FA3 split selection](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/heuristics.h)
and [FlashInfer planning](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh),
not an equivalent cost model.

## Automatic eligibility

The supported `(query heads, KV heads, head dimension)` geometries are
`(32,8,128)`, `(24,8,128)`, `(24,4,256)`, `(16,2,128)`, and `(16,2,256)`.
Kernel block16 is supported for each; block32 is additionally supported for
`(24,4,256)` and `(16,2,256)`. The existing scope is not inferred from a model-name
list.
Geometry and kernel-page admission are checked at the native dispatch gate
before occupancy planning, including when a private test forces the producer.
One C++ table owns this domain. Numerical and routing tests read that table
through `_gqa_decode_config_for_test()` instead of maintaining another list;
golden threshold values and unsupported-input expectations remain independent.

Let `C` be the detected GPU core count, `Q` the query-head count, and `L_i`
each request's KV length including its current decode token. The producer is
used when `Q * sum(floor(L_i / 256)) >= 33 * C`, an empirical budget of 33 SIMD
groups per core; otherwise the established path runs. Flooring each length
separately avoids counting partial tails as complete work or treating short
requests as copies of the longest.

The same budget applies across batch sizes. The GQA planner does not reject a
batch solely because its unsplit query-head grid exceeds a core-count
threshold. The established per-token fallback retains its own split-KV policy.

A single request starts at `256 * ceil(33 * C / Q)` tokens. On a 40-core GPU:

| Q/KV/head dimension | Start |
|---|---:|
| 32/8/128 | 10,752 |
| 24/8/128 or 24/4/256 | 14,080 |
| 16/2/128 or 256 | 21,248 |

These values scale with the detected core count; they are not per-device or
per-model context tables in the implementation, and the budget has no
per-model exceptions. Core-count scaling does not establish cross-device
performance or guarantee a speedup at every boundary. Two `32/8/128` requests
on that GPU start at 5,376 tokens each. Lengths `[1, 10752]` are admitted too:
the long request supplies enough complete partitions and the short one none.

Only complete partitions count toward admission. The producer, temporary
buffers and reducer still use `ceil(KV_length / 256)` so the final partial
partition is processed. The first ordinary decode layer builds an immutable
CPU length plan once per forward, shared across layers and KV groups. A single
pass records the request count, maximum length and complete-partition total.
Each native layer copies this fixed-size value and applies its
geometry's policy without re-copying or re-scanning the request list. A new
forward gets a new plan; deferred execution retains the original value.
Single- and multi-request selection use the same planner. There is no GPU
readback, startup benchmark or per-request performance probe.
`gqa_decode_partition_size` exposes the default decision for tests, 256 or
zero.
`gqa_decode_batch_partition_size` takes a list of per-request lengths;
`max_seq_len` optionally supplies an allocation upper bound.
These queries check geometry, work and the scratch budget, not all dispatch
conditions. Direct primitive callers can pass `gqa_length_plan` from
`gqa_decode_length_plan(context_lens)`; it must agree with GPU `seq_lens`.

Kernel block size is the view after hybrid-cache translation. The largest
supported size that evenly divides the scheduler page is selected: an 800- or
1056-token page selects block32, while a 784- or 528-token page selects block16.
Translated page IDs address their kernel-sized subpages even when upstream
K/V storage remains unreshaped.

The GQA producer walks pages rather than tokens. Lane 0 preloads the next
valid block-table entry and `simd_shuffle` broadcasts it within each SIMD
group. A 4-token inner step computes independent QK dots and combines their
online-softmax update; a scalar remainder handles the final 1-3 tokens.
Every tier reads K/V directly, without threadgroup staging or barriers.
Online-softmax state stays in registers.

The producer writes the same log2-space `(max, exp-sum)` plus
epsilon-normalized `tmp_out` contract as split-KV. The shared
`paged_attention_v2_reduce` therefore merges GQA partials unchanged:
the GQA producer does not require a separate reduction algorithm.

Every eligible call additionally requires:

- Ordinary decode rows with exactly one query row per request: either a
  whole pure-decode batch, or the leading decode rows that the native split
  separates from a mixed batch (see below). One request in a pure-decode
  batch retains the existing `num_decode_requests=1` or omitted-count
  convention. Multiple requests require matching `num_decode_requests` and
  `num_decode_tokens`, plus their CPU `gqa_length_plan` metadata. A mixed-batch
  decode prefix requires a plan describing exactly its rows.
- A verification window of at most 1.
- Matching FP16/BF16 query, key-cache and value-cache types, or an FP16/BF16
  query over a TurboQuant cache (int8/uint8 K, packed uint8 V). TurboQuant
  takes the same work budget. A TurboQuant decode batch
  reaches this path only as a whole batch; the compressed fallback rows of a
  bounded TurboQuant prefill keep the established kernels.
- At most **512 MiB total GQA scratch per attention call**. With multi-request
  length metadata, padded rows use compact scratch:
  `Q * sum(ceil(length_i / P)) * (2 * head_dim + 8) + 8 * (B + 1)` bytes for
  the FP16/BF16 partial output, two FP32 statistics and INT64 partition offsets.
  Calls with no padding or no multi-request metadata retain the rectangular layout:
  `B * Q * ceil(max_seq_len / P) * (2 * head_dim + 8)` bytes.
  Larger calls fall back before allocation. This ceiling covers the measured
  serving windows; it is not a reservation or a bound on total process memory.
- A kernel page size allowed above and sufficient reducer shared memory:
  aligned dynamic statistics plus the compiled pipeline's static allocation.
  Apple GPUs' 32 KiB threadgroup limit fits 4,088 partitions, so `max_seq_len`
  can be at most 1,046,528 tokens.
  This is checked before allocating GQA scratch or encoding the producer;
  an oversized `max_seq_len` allocation bound falls back even when the actual
  KV lengths pass the work budget. The reducer reuses the checked pipeline.
- No attention sinks, logit soft-capping or sliding window.
- A known, positive GPU core count.

Other calls, including mixed batches without a decode-prefix plan, missing
batch lengths, expanded verification rows and unknown core counts, use the
established attention family. Model context limits, cache capacity and
primitive resource limits still apply. The 16/4/256 geometry remains
excluded from default routing.

## Disable switch

Set `VLLM_METAL_DISABLE_GQA_DECODE=1` before starting the server to keep
eligible requests on the established path. This switch provides an A/B and
operational fallback; it cannot enable an otherwise ineligible call.

The switch is captured once when each forward's `PagedAttentionContext` is
created and shared by its layers. The native module reports its public ABI
through one `paged_attention_capabilities()` query, also cached per forward.
Kernel availability is distinct from a supported production GQA route.
The `gqa_disabled` keyword is sent only when disabling GQA on a native build
that reports control support. Pre-GQA builds receive no new keyword; an
unrecognized GQA build without disable support requires a rebuild rather than
silently ignoring the switch. The structured query is part of the required
extension ABI; an artifact too old to export it is unsupported and fails at
the query instead of probing older entry points.
The `gqa_length_plan` capability advertises the per-forward plan. Builds
without it keep their existing batch routing.

## Routing contract

Kernel dtype, page layout, query-row shape, feature and resource checks determine
whether an operation can run. Default admission is a separate measured policy:
ordinary decode, the listed geometries and the work budget.
The C++ boundary enforces both, including for direct primitive callers.

The native planner consumes post-append KV lengths, query/KV head geometry
and detected core count; its result is fallback or the GQA producer. For a whole
ordinary decode batch, SDPA forwards the scheduler's existing CPU lengths.
The primitive captures its own copy and includes it in lazy-node equivalence.
Direct callers must supply lengths consistent with `seq_lens`; shape and
positive-length bounds are checked eagerly, without reading GPU array data.

The CPU length plan also captures per-request partition offsets, shared
immutably across layers.
When rows would need padding, the producer and reducer address tightly packed
per-request partials using these offsets. Rows that all fill the dispatch stride
retain their existing layout without an offset buffer. The producer grid and
reducer shared memory remain sized by the maximum partition count; each producer still skips
partitions beyond its own KV length, and each reducer reads only that request's
valid partials in the same order. Compact scratch saves allocation space without
changing partition selection or removing empty producer workgroups.

Mixed batches use an explicit decode sub-batch. The native split separates the
leading ordinary decode rows from the prefill kernel (#851 for tiled prefill,
and NAX on M5) when the batch carries decode metadata and the decode rows hold
at least 512 tokens of KV work. `sdpa_forward` passes a `GqaDecodeLengthPlan`
built from only those rows' context lengths (capability
`gqa_mixed_decode_plan`); the work is the plan's exact sum of those lengths.
Without a plan, the longest decode row stands in for it as a lower bound. The
prefix call then treats rows and sequences `0..D-1` as a pure decode batch: the
GQA grid, reducer and scratch use `D` requests and the plan's maximum length,
not the batch-wide count or `max_seq_len`. Without that plan, or with GQA
disabled, the prefix keeps the per-token kernel, whose split-KV partitions are
likewise bounded by `max_decode_context_len` rather than `max_seq_len` (which
also covers the prefill rows and may be an allocation bound). Admission counts
only the decode rows: a long prefill chunk never admits a short decode prefix.
The binding rejects a prefix plan whose request count differs from the decode
rows or whose maximum exceeds `max_decode_context_len`.

## Validation

The GQA tests cover numerical agreement, production routing and fallbacks,
resource limits, ragged batches, and cache updates under lazy evaluation.

```bash
python -m pytest tests/test_gqa_*.py -m "not slow"
python -m pytest tests/test_gqa_paged_decode.py tests/test_gqa_decode_routing.py -m slow
```

Forced-kernel tests bypass performance admission. Check `last_paged_dispatch()`
when testing default routing; numerical agreement alone does not show that GQA ran.

Dispatch diagnostics are disabled by default. The ordinary dispatch path
reads one relaxed atomic flag and returns without writing diagnostic state.
Serial tests and isolated benchmark workers explicitly enable recording:

```python
import mlx.core as mx
from vllm_metal.metal import get_ops

ops = get_ops()
mx.synchronize()  # Toggle only while evaluation is idle.
previous = ops._set_paged_dispatch_diagnostics(True)
try:
    # Run and evaluate the attention operation in this process.
    ...
    family = ops.last_paged_dispatch()
    partition = ops.last_gqa_partition_size()
    requests = ops.last_gqa_num_requests()
finally:
    mx.synchronize()
    ops._set_paged_dispatch_diagnostics(previous)
```

Enable recording in the worker that executes attention, before its measured
requests; enabling it only in the HTTP client does not observe the worker.
Both toggles clear the last observation. While disabled, the getters return
an empty family and zero partition/request count. Recording changes neither routing nor
attention computation. These are process-wide diagnostics, not per-request
telemetry for concurrent serving. Evaluate the operation before reading its
record, because MLX builds graphs lazily.

For performance validation, rebuild native artifacts from the tested
revision and use the real `vllm serve` process topology. Record the model,
runtime versions, hardware, warmup, KV lengths, repetitions, and background
GPU activity; compare enabled and disabled arms with actual worker-side
dispatch checks. Separate primitive timing from HTTP decode throughput,
and keep noisy measurements visible. The
[macOS benchmarking guide](benchmarking-macos.md)
explains why in-process engine measurements and short probes are not
interchangeable with serving results.

## Design discussions

See [GQA routing RFC #1328](https://github.com/vllm-project/vllm-metal/issues/1328)
for proposed extensions to routing and backend support.
