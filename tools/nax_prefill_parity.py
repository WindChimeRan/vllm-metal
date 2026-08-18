#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manual M5 parity check for NAX versus tiled paged prefill attention.

Fails when NAX is unavailable so a release check cannot silently skip it.
Run with ``python tools/nax_prefill_parity.py``.
"""

from __future__ import annotations

import math
import sys

import mlx.core as mx

from vllm_metal.metal import get_ops

SEQS = [(129, 130), (1, 999), (64, 0), (1, 333)]
Q_HEADS = 32
KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 16
SOFTCAP = 30.0
SLIDING_WINDOW = 64
MAX_ABS_ERROR = 2e-2


def _build_case():
    mx.random.seed(3)
    totals = [q + c for q, c in SEQS]
    blocks_per = [(t + BLOCK_SIZE - 1) // BLOCK_SIZE for t in totals]
    num_blocks = sum(blocks_per) + 1  # block 0 is padding

    key_cache = mx.random.normal((num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_SIZE)).astype(
        mx.bfloat16
    )
    value_cache = mx.random.normal(
        (num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_SIZE)
    ).astype(mx.bfloat16)
    query = (
        mx.random.normal((sum(q for q, _ in SEQS), Q_HEADS, HEAD_SIZE)) * 0.5
    ).astype(mx.bfloat16)

    # Non-contiguous physical pages exercise the block-table gather.
    block_ids = [*range(1, num_blocks, 2), *range(2, num_blocks, 2)]
    rows, offset = [], 0
    for count in blocks_per:
        rows.append(block_ids[offset : offset + count])
        offset += count
    max_blocks = max(blocks_per)
    block_tables = mx.array(
        [row + [0] * (max_blocks - len(row)) for row in rows], dtype=mx.int32
    )
    seq_lens = mx.array(totals, dtype=mx.int32)
    cu_seqlens = mx.array(
        [0, *mx.cumsum(mx.array([q for q, _ in SEQS], dtype=mx.int32)).tolist()],
        dtype=mx.int32,
    )
    sinks = (mx.random.normal((Q_HEADS,)) * 0.5).astype(mx.float32)
    mx.eval(key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens, sinks)
    return key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens, sinks


def _run(ops, case, *, enabled: bool) -> mx.array:
    key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens, sinks = case
    ops.set_nax_enabled(enabled)
    out = mx.array(0)
    ops.paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        KV_HEADS,
        HEAD_SIZE**-0.5,
        SOFTCAP,
        block_tables,
        seq_lens,
        cu_seqlens,
        BLOCK_SIZE,
        max(int(x) for x in seq_lens.tolist()),
        SLIDING_WINDOW,
        out,
        sinks=sinks,
    )
    mx.eval(out)
    return out.astype(mx.float32)


def main() -> int:
    ops = get_ops()
    if not (ops.nax_supported() and ops.nax_ready()):
        print(
            "NAX is unavailable; run this check on an M5 with the NAX metallib.",
            file=sys.stderr,
        )
        return 2

    case = _build_case()
    try:
        nax = _run(ops, case, enabled=True)
        tiled = _run(ops, case, enabled=False)
    finally:
        ops.set_nax_enabled(True)

    mx.eval(nax, tiled)
    max_abs_error = mx.abs(nax - tiled).max().item()
    print(f"NAX vs tiled max absolute error: {max_abs_error:.6g}")
    if not math.isfinite(max_abs_error) or max_abs_error > MAX_ABS_ERROR:
        print(f"PARITY FAIL: limit={MAX_ABS_ERROR:.3g}", file=sys.stderr)
        return 1
    print("PARITY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
