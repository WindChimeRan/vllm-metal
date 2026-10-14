# SPDX-License-Identifier: Apache-2.0
"""Tests for the paged attention primitive.

Run with:
    python -m pytest tests/test_primitive_and_donation.py -v -s
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn
from vllm_metal.metal import get_ops

# ── Shared fixtures ──────────────────────────────────────────────────────────

NUM_KV_HEADS_CASES = [2, 4]
NUM_QUERY_HEADS_CASES = [4, 8]  # must be divisible by corresponding kv heads
HEAD_SIZE = 128
BLOCK_SIZE = 16
DTYPE = mx.float16


def _make_cache_and_inputs(
    num_blocks: int,
    num_kv_heads: int,
    num_query_heads: int,
    seq_lens: list[tuple[int, int]],
    *,
    head_size: int = HEAD_SIZE,
    dtype: mx.Dtype = DTYPE,
):
    """Build a populated cache and matching query/metadata tensors."""
    block_size = BLOCK_SIZE
    num_seqs = len(seq_lens)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    total_q = sum(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    key_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    query = mx.random.normal(shape=(total_q, num_query_heads, head_size)).astype(dtype)

    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = mx.random.randint(
        0, num_blocks, shape=(num_seqs, max_blocks_per_seq)
    ).astype(mx.int32)

    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)
    cu_seqlens_q = mx.cumsum(mx.array([0] + query_lens, dtype=mx.int32))

    mx.eval(key_cache, value_cache, query, block_tables, kv_lens_arr, cu_seqlens_q)

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "num_kv_heads": num_kv_heads,
        "scale": scale,
        "block_tables": block_tables,
        "kv_lens_arr": kv_lens_arr,
        "cu_seqlens_q": cu_seqlens_q,
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "max_kv_len": max_kv_len,
    }


# ── 1. Primitive correctness tests ──────────────────────────────────────────


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1), (1, 128), (1, 2048)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(4, 4), (8, 2)],
)
@pytest.mark.parametrize("sliding_window", [-1, 128])
@pytest.mark.parametrize("num_blocks", [256])
def test_primitive_vs_reference_decode(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    num_blocks: int,
    sliding_window: int,
) -> None:
    """paged_attention_primitive matches the pure-MLX reference (decode)."""
    mx.random.seed(0)
    num_query_heads, num_kv_heads = num_heads
    d = _make_cache_and_inputs(num_blocks, num_kv_heads, num_query_heads, seq_lens)
    num_decode_requests = len(seq_lens)

    ops = get_ops()
    out = mx.array(0)
    ops.paged_attention_primitive(
        d["query"],
        d["key_cache"],
        d["value_cache"],
        d["num_kv_heads"],
        d["scale"],
        0.0,  # softcap
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        BLOCK_SIZE,
        d["max_kv_len"],
        sliding_window,
        out,
        num_decode_requests=num_decode_requests,
        num_decode_tokens=num_decode_requests,
        max_decode_context_len=max(kv_len for _, kv_len in seq_lens),
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=d["query"],
        key_cache=d["key_cache"],
        value_cache=d["value_cache"],
        query_lens=d["query_lens"],
        kv_lens=d["kv_lens"],
        block_tables=np.array(d["block_tables"]),
        scale=d["scale"],
        sliding_window=sliding_window if sliding_window >= 0 else None,
    )
    mx.eval(ref)

    np.testing.assert_allclose(
        np.array(out),
        np.array(ref),
        atol=1.5e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],
        [(1, 523), (1, 37), (1, 2011)],
        # Decode prefix crosses the tiled kernel's 32-row block boundary.
        [(1, 4096)] + [(1, 64)] * 32 + [(32, 64)],
        # Prefill-only: all q_len > 1, guarantees tiled kernel dispatch
        # (total_q_tokens > num_seqs).
        [(8, 128), (16, 256)],
        [(32, 512), (64, 1024)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(4, 4), (8, 2)],
)
@pytest.mark.parametrize("sliding_window", [-1, 100])
@pytest.mark.parametrize("num_blocks", [256])
def test_primitive_vs_reference_varlen(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    num_blocks: int,
    sliding_window: int,
    force_tiled_prefill,
) -> None:
    """paged_attention_primitive matches reference for mixed prefill+decode."""
    mx.random.seed(0)
    num_query_heads, num_kv_heads = num_heads
    d = _make_cache_and_inputs(num_blocks, num_kv_heads, num_query_heads, seq_lens)
    num_decode_requests = 0
    for query_len, _ in seq_lens:
        if query_len != 1:
            break
        num_decode_requests += 1
    max_decode_context_len = max(
        (seq_lens[i][1] for i in range(num_decode_requests)), default=0
    )

    ops = get_ops()
    out = mx.array(0)
    ops.paged_attention_primitive(
        d["query"],
        d["key_cache"],
        d["value_cache"],
        d["num_kv_heads"],
        d["scale"],
        0.0,  # softcap
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        BLOCK_SIZE,
        d["max_kv_len"],
        sliding_window,
        out,
        num_decode_requests=num_decode_requests,
        num_decode_tokens=num_decode_requests,
        max_decode_context_len=max_decode_context_len,
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=d["query"],
        key_cache=d["key_cache"],
        value_cache=d["value_cache"],
        query_lens=d["query_lens"],
        kv_lens=d["kv_lens"],
        block_tables=np.array(d["block_tables"]),
        scale=d["scale"],
        sliding_window=sliding_window if sliding_window >= 0 else None,
    )
    mx.eval(ref)

    np.testing.assert_allclose(
        np.array(out),
        np.array(ref),
        atol=1.5e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "decode_context,head_size,split_decode",
    [
        (512, 128, False),
        (4095, 128, False),
        (4096, 128, True),
        (4096, 256, True),
        (4096, 512, True),
    ],
)
def test_mixed_batch_dispatches_decode_by_context(
    decode_context: int,
    head_size: int,
    split_decode: bool,
    force_tiled_prefill,
) -> None:
    """Only long decode contexts pay for a separate kernel dispatch."""
    decode_rows = 33
    seq_lens = [(1, decode_context)] + [(1, 64)] * (decode_rows - 1) + [(32, 64)]
    mx.random.seed(0)
    d = _make_cache_and_inputs(512, 2, 8, seq_lens, head_size=head_size)
    ops = get_ops()

    def run_attention(
        query: mx.array,
        block_tables: mx.array,
        kv_lens: mx.array,
        cu_seqlens: mx.array,
        decode_count: int = 0,
        max_context: int = 0,
    ) -> np.ndarray:
        out = mx.array(0)
        ops.paged_attention_primitive(
            query,
            d["key_cache"],
            d["value_cache"],
            d["num_kv_heads"],
            d["scale"],
            0.0,
            block_tables,
            kv_lens,
            cu_seqlens,
            BLOCK_SIZE,
            d["max_kv_len"],
            -1,
            out,
            num_decode_requests=decode_count,
            num_decode_tokens=decode_count,
            max_decode_context_len=max_context,
        )
        mx.eval(out)
        return np.array(out)

    mixed = run_attention(
        d["query"],
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        decode_rows,
        decode_context,
    )
    whole_tiled = run_attention(
        d["query"], d["block_tables"], d["kv_lens_arr"], d["cu_seqlens_q"]
    )
    pure_decode = run_attention(
        d["query"][:decode_rows],
        d["block_tables"][:decode_rows],
        d["kv_lens_arr"][:decode_rows],
        mx.arange(decode_rows + 1, dtype=mx.int32),
        decode_rows,
        decode_context,
    )

    expected_decode = pure_decode if split_decode else whole_tiled[:decode_rows]
    np.testing.assert_array_equal(mixed[:decode_rows], expected_decode)
    np.testing.assert_array_equal(mixed[decode_rows:], whole_tiled[decode_rows:])


@pytest.mark.slow
def test_sliding_window_bounds_tiled_prefill_work(force_tiled_prefill) -> None:
    """A 1024-token window avoids most KV tiles of an 8K prefill."""
    seq_len = 8192
    window = 1024
    repeats = 5
    minimum_speedup = 2.0
    num_kv_heads = 2
    num_query_heads = 4
    d = _make_cache_and_inputs(
        seq_len // BLOCK_SIZE,
        num_kv_heads,
        num_query_heads,
        [(seq_len, seq_len)],
        head_size=64,
    )
    ops = get_ops()

    def run(sliding_window: int) -> None:
        out = mx.array(0)
        ops.paged_attention_primitive(
            d["query"],
            d["key_cache"],
            d["value_cache"],
            d["num_kv_heads"],
            d["scale"],
            0.0,
            d["block_tables"],
            d["kv_lens_arr"],
            d["cu_seqlens_q"],
            BLOCK_SIZE,
            d["max_kv_len"],
            sliding_window,
            out,
        )
        mx.eval(out)

    def median_runtime(sliding_window: int) -> float:
        run(sliding_window)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            run(sliding_window)
            samples.append(time.perf_counter() - start)
        return sorted(samples)[len(samples) // 2]

    full = median_runtime(-1)
    windowed = median_runtime(window)
    speedup = full / windowed
    assert speedup >= minimum_speedup, f"sliding-window speedup was only {speedup:.2f}x"


# ── 1b. Tiled-kernel head-size coverage ─────────────────────────────────────


@pytest.mark.parametrize("head_size", [64, 96, 128])
def test_tiled_kernel_head_size_parity(head_size: int, force_tiled_prefill) -> None:
    """Tiled prefill kernel matches reference for every routed head size.

    can_use_tiled_kernel() routes head sizes {64, 96, 128} to the BQ=32
    flash kernel; each is a separately compiled template specialization
    (TD = HEAD_SIZE / 8 differs).  The varlen test only exercises 128, so
    this adds one prefill shape per head size to keep CI parity 1:1 with
    can_use_tiled_kernel().
    """
    mx.random.seed(0)
    # Prefill-only (q_len > 1 for every seq) guarantees tiled dispatch.
    seq_lens = [(32, 512), (64, 1024)]
    d = _make_cache_and_inputs(256, 2, 8, seq_lens, head_size=head_size)

    ops = get_ops()
    out = mx.array(0)
    ops.paged_attention_primitive(
        d["query"],
        d["key_cache"],
        d["value_cache"],
        d["num_kv_heads"],
        d["scale"],
        0.0,  # softcap
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        BLOCK_SIZE,
        d["max_kv_len"],
        -1,  # sliding_window
        out,
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=d["query"],
        key_cache=d["key_cache"],
        value_cache=d["value_cache"],
        query_lens=d["query_lens"],
        kv_lens=d["kv_lens"],
        block_tables=np.array(d["block_tables"]),
        scale=d["scale"],
        sliding_window=None,
    )
    mx.eval(ref)

    np.testing.assert_allclose(
        np.array(out),
        np.array(ref),
        atol=1.5e-2,
        rtol=1e-2,
    )
