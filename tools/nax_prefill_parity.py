#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manual M5 parity check for NAX versus tiled paged prefill attention.

Deliberately not wired into CI: no GitHub runner has an M5, and a check that
silently skips itself is worse than no check. Run it on an M5 before touching
``pagedattention_nax.metal`` or the NAX dispatch::

    python tools/nax_prefill_parity.py

Covers every compiled specialization -- {float16, bfloat16} x head_size
{64, 96, 128, 256, 512} x block_size {8, 16, 32} -- in both masking regimes:

``fast``
    Softcap off, no sliding window, sinks off. Reaches the ``tile_no_mask``
    arm (where ``min_q_abs_sg`` and the ``row_lim >= 16`` guard are
    load-bearing) and the ``_sk0`` pipeline, whose softmax init is the one that
    can end a row with a zero denominator.
``masked``
    Softcap, sliding window and sinks all on: the per-element mask arm and the
    ``_sk1`` pipeline.

Block size matters beyond the kernel name: at 8 the four half-row bases of a
32-token KV tile land on four different pages, at 32 they collapse onto one.
Head sizes 64/96 (TD=4/6) skip the mid-loop scheduling barrier. Head sizes
128/256/512 (TD=8/16/32) split the PV loop at its midpoint; their output
accumulators hold 64/128/256 fp32 values per thread, respectively.

Every case is checked for *engagement*, not just agreement. NAX accumulates
through the tensor unit with ``relaxed_precision``, so a bitwise-identical
result means the batch never reached NAX -- eligibility rejected it and the
comparison was vacuous.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import mlx.core as mx

from vllm_metal.metal import get_ops

# (query tokens, context tokens) per sequence. One-token rows exercise decode
# passengers in a mixed prefill batch.
SEQS = [(129, 130), (1, 999), (64, 0), (1, 333)]
Q_HEADS = 32
KV_HEADS = 8

SOFTCAP = 30.0
SLIDING_WINDOW = 64

# NAX truncates the fp32 P operand of PV (relaxed_precision) and accumulates in
# a different order than the tiled kernel, so exact agreement is not the bar --
# but output-dtype rounding is. Divergence is measured in ULPs of the output
# dtype at the batch's peak magnitude, which keeps one bound meaningful for both
# fp16 and bf16 (a plain absolute bound loose enough for bf16 is 8x too loose
# for fp16, and a per-element relative error is meaningless where attention
# outputs approach zero).
#
# Measured across the full matrix on an M5 Pro (macOS 26.6): worst case
# 0.88 ULP max / 0.022 ULP mean. The mean is the one that catches a mask or
# geometry bug -- those shift whole rows, while rounding noise stays scattered.
_DTYPE_EPS = {mx.float16: 2**-10, mx.bfloat16: 2**-7}
MAX_ULP = 2.0
MEAN_ULP = 0.05


@dataclass(frozen=True)
class Case:
    dtype: mx.Dtype
    head_size: int
    block_size: int
    regime: str  # "fast" (tile_no_mask + _sk0) | "masked" (mask arm + _sk1)

    @property
    def softcap(self) -> float:
        return 0.0 if self.regime == "fast" else SOFTCAP

    @property
    def sliding_window(self) -> int:
        return -1 if self.regime == "fast" else SLIDING_WINDOW

    @property
    def use_sinks(self) -> bool:
        return self.regime == "masked"

    def __str__(self) -> str:
        return (
            f"{str(self.dtype).removeprefix('mlx.core.'):9s} "
            f"hs={self.head_size:<3d} bs={self.block_size:<2d} {self.regime}"
        )


def _cases() -> list[Case]:
    """The full instantiated matrix: instantiate_paged_attention_nax_all emits
    {half, bfloat16_t} x hs{64, 96, 128, 256, 512} x bs{8, 16, 32}, and the
    dispatcher picks a separate pipeline per sinks constant."""
    return [
        Case(dtype, head_size, block_size, regime)
        for dtype in (mx.float16, mx.bfloat16)
        for head_size in (64, 96, 128, 256, 512)
        for block_size in (8, 16, 32)
        for regime in ("fast", "masked")
    ]


def _build_case(case: Case):
    mx.random.seed(3)
    block_size, head_size = case.block_size, case.head_size
    totals = [q + c for q, c in SEQS]
    blocks_per = [(t + block_size - 1) // block_size for t in totals]
    num_blocks = sum(blocks_per) + 1  # block 0 is padding

    shape = (num_blocks, block_size, KV_HEADS, head_size)
    key_cache = mx.random.normal(shape).astype(case.dtype)
    value_cache = mx.random.normal(shape).astype(case.dtype)
    query = (
        mx.random.normal((sum(q for q, _ in SEQS), Q_HEADS, head_size)) * 0.5
    ).astype(case.dtype)

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
    sinks = (
        (mx.random.normal((Q_HEADS,)) * 0.5).astype(mx.float32)
        if case.use_sinks
        else None
    )
    mx.eval(key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens)
    if sinks is not None:
        mx.eval(sinks)
    return key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens, sinks


def _run(ops, case: Case, built, *, enabled: bool) -> mx.array:
    key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens, sinks = built
    ops.set_nax_enabled(enabled)
    out = mx.array(0)
    ops.paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        KV_HEADS,
        case.head_size**-0.5,
        case.softcap,
        block_tables,
        seq_lens,
        cu_seqlens,
        case.block_size,
        max(int(x) for x in seq_lens.tolist()),
        case.sliding_window,
        out,
        sinks=sinks,
    )
    mx.eval(out)
    return out.astype(mx.float32)


def _check(ops, case: Case) -> tuple[bool, str]:
    built = _build_case(case)
    nax = _run(ops, case, built, enabled=True)
    tiled = _run(ops, case, built, enabled=False)
    mx.eval(nax, tiled)

    delta = mx.abs(nax - tiled)
    abs_err = delta.max().item()
    ulp = _DTYPE_EPS[case.dtype] * max(mx.abs(tiled).max().item(), 1.0)
    max_ulp = abs_err / ulp
    mean_ulp = delta.mean().item() / ulp

    if abs_err == 0.0:
        # Both runs took the same kernel: nax_eligible() or the batch-shape
        # guard rejected this case, so nothing about NAX was tested.
        return False, "NAX DID NOT RUN (identical output; check eligibility)"
    if not math.isfinite(abs_err):
        return False, f"non-finite output (abs={abs_err})"
    detail = f"max={max_ulp:.2f}ulp mean={mean_ulp:.4f}ulp"
    if max_ulp > MAX_ULP or mean_ulp > MEAN_ULP:
        return False, f"{detail} exceeds bound"
    return True, detail


def main() -> int:
    ops = get_ops()
    if not (ops.nax_supported() and ops.nax_ready()):
        print(
            "NAX is unavailable; run this check on an M5 with the NAX metallib.",
            file=sys.stderr,
        )
        return 2

    failures = []
    try:
        for case in _cases():
            ok, detail = _check(ops, case)
            print(f"{'ok  ' if ok else 'FAIL'} {case}  {detail}")
            if not ok:
                failures.append(f"{case}: {detail}")
    finally:
        ops.set_nax_enabled(True)

    if failures:
        print(
            f"\nPARITY FAIL ({len(failures)} case(s), "
            f"limits max={MAX_ULP:.3g}ulp mean={MEAN_ULP:.3g}ulp):",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"\nPARITY PASS ({len(_cases())} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
