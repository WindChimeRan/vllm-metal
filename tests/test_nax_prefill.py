# SPDX-License-Identifier: Apache-2.0
"""NAX and tiled prefill parity against a gathered fp32 SDPA reference.

The relaxed-precision PV matmul is not bitwise identical to tiled attention.
Tests skip when the optional NAX library is unavailable.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from vllm_metal.metal import get_ops


def _nax_ready() -> bool:
    return bool(get_ops().nax_ready())


pytestmark = pytest.mark.skipif(
    not _nax_ready(),
    reason="NAX prefill kernels unavailable on this machine",
)


def _build_case(seqs, qh, kvh, d, bs, dtype, seed):
    """seqs: list of (query_len, context_len); cache holds ctx+q tokens/seq."""
    mx.random.seed(seed)
    totals = [q + c for q, c in seqs]
    blocks_per = [(t + bs - 1) // bs for t in totals]
    num_blocks = sum(blocks_per) + 1  # block 0 unused (padding)
    kc = mx.random.normal((num_blocks, bs, kvh, d)).astype(dtype)
    vc = mx.random.normal((num_blocks, bs, kvh, d)).astype(dtype)
    total_q = sum(q for q, _ in seqs)
    q = (mx.random.normal((total_q, qh, d)) * 0.5).astype(dtype)
    # Interleave physical page IDs so every case exercises block-table gathers.
    block_ids = [*range(1, num_blocks, 2), *range(2, num_blocks, 2)]
    bt_rows, nxt = [], 0
    for nb in blocks_per:
        bt_rows.append(block_ids[nxt : nxt + nb])
        nxt += nb
    max_nb = max(blocks_per)
    bt = mx.array([r + [0] * (max_nb - len(r)) for r in bt_rows], dtype=mx.int32)
    seq_lens = mx.array(totals, dtype=mx.int32)
    cu = [0]
    for ql, _ in seqs:
        cu.append(cu[-1] + ql)
    cu = mx.array(cu, dtype=mx.int32)
    mx.eval(kc, vc, q, bt, seq_lens, cu)
    return kc, vc, q, bt, seq_lens, cu, bt_rows, totals


def _reference(q, kc, vc, seqs, bt_rows, qh, kvh, d, bs, scale, softcap, window, sinks):
    """Gathered fp32 SDPA with prefix-causal + window + softcap + sinks."""
    flat_k = kc.reshape(-1, kvh, d).astype(mx.float32)
    flat_v = vc.reshape(-1, kvh, d).astype(mx.float32)
    group = qh // kvh
    outs, q_off = [], 0
    for (ql, ctx), row in zip(seqs, bt_rows, strict=True):
        total = ql + ctx
        slots = mx.array([row[p // bs] * bs + p % bs for p in range(total)])
        kh = mx.repeat(flat_k[slots].transpose(1, 0, 2), group, axis=0)
        vh = mx.repeat(flat_v[slots].transpose(1, 0, 2), group, axis=0)
        qs = q[q_off : q_off + ql].astype(mx.float32).transpose(1, 0, 2)
        scores = (qs @ kh.transpose(0, 2, 1)) * scale
        if softcap > 0:
            scores = softcap * mx.tanh(scores / softcap)
        pos = mx.arange(total)[None, None, :]
        qpos = (ctx + mx.arange(ql))[None, :, None]
        mask = pos > qpos
        if window >= 0:
            mask = mask | (pos < qpos + 1 - window)
        scores = mx.where(mask, mx.array(-1e30), scores)
        if sinks is not None:
            sk = mx.broadcast_to(sinks[:, None, None].astype(mx.float32), (qh, ql, 1))
            scores = mx.concatenate([sk, scores], axis=-1)
            attn = mx.softmax(scores, axis=-1)[..., 1:]
        else:
            attn = mx.softmax(scores, axis=-1)
        outs.append((attn @ vh).transpose(1, 0, 2))
        q_off += ql
    return mx.concatenate(outs, axis=0)


CASES = {
    "single": {"seqs": [(200, 0)]},
    "block_boundary_257": {"seqs": [(257, 0)]},
    "tiny_q3": {"seqs": [(3, 0)]},
    "row_spill_65": {"seqs": [(65, 0)]},
    "chunked_prefill": {"seqs": [(70, 91)]},
    "chunked_unaligned_ctx": {"seqs": [(64, 33)]},
    "mixed_prefill_decode": {"seqs": [(150, 0), (1, 77), (1, 300), (33, 12)]},
    "d64": {"seqs": [(190, 30)], "d": 64},
    "bs8": {"seqs": [(180, 60)], "bs": 8},
    "bs32": {"seqs": [(180, 60)], "bs": 32},
    "fp16": {"seqs": [(160, 0)], "dtype": mx.float16},
    "mha": {"seqs": [(130, 20)], "qh": 8, "kvh": 8},
    "mqa": {"seqs": [(90, 10)], "qh": 32, "kvh": 1},
    "sliding_window": {"seqs": [(200, 100)], "window": 64},
    "window_lt_tile": {"seqs": [(100, 300)], "window": 5},
    "softcap": {"seqs": [(150, 0)], "softcap": 30.0},
    "sinks": {"seqs": [(140, 25)], "use_sinks": True},
    "sinks_window": {"seqs": [(140, 25)], "use_sinks": True, "window": 40},
    "stress_mixed": {"seqs": [(511, 130), (1, 999), (64, 0), (2, 2), (300, 1)]},
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_nax_prefill_matches_reference(name):
    cfg = CASES[name]
    seqs = cfg["seqs"]
    qh = cfg.get("qh", 32)
    kvh = cfg.get("kvh", 8)
    d = cfg.get("d", 128)
    bs = cfg.get("bs", 16)
    dtype = cfg.get("dtype", mx.bfloat16)
    softcap = cfg.get("softcap", 0.0)
    window = cfg.get("window", -1)
    use_sinks = cfg.get("use_sinks", False)

    kc, vc, q, bt, seq_lens, cu, bt_rows, totals = _build_case(
        seqs, qh, kvh, d, bs, dtype, seed=3
    )
    scale = d**-0.5
    sinks = None
    if use_sinks:
        sinks = (mx.random.normal((qh,)) * 0.5).astype(mx.float32)
        mx.eval(sinks)

    ops = get_ops()

    def run_primitive() -> mx.array:
        out = mx.array(0)
        ops.paged_attention_primitive(
            q,
            kc,
            vc,
            kvh,
            scale,
            softcap,
            bt,
            seq_lens,
            cu,
            bs,
            max(totals),
            window,
            out,
            sinks=sinks,
        )
        return out

    ops.set_nax_enabled(True)
    try:
        nax = run_primitive()
        mx.eval(nax)
        ops.set_nax_enabled(False)
        tiled = run_primitive()
        mx.eval(tiled)
    finally:
        ops.set_nax_enabled(True)
    ref = _reference(
        q,
        kc,
        vc,
        seqs,
        bt_rows,
        qh,
        kvh,
        d,
        bs,
        scale,
        softcap,
        window,
        sinks,
    )
    nax_f = nax.reshape(-1, qh, d).astype(mx.float32)
    tiled_f = tiled.reshape(-1, qh, d).astype(mx.float32)
    mx.eval(nax_f, tiled_f, ref)

    nax_err = mx.abs(nax_f - ref).max().item()
    tiled_err = mx.abs(tiled_f - ref).max().item()
    # Allow relaxed-precision PV headroom relative to the tiled error.
    assert nax_err <= max(2.5 * tiled_err, 2e-2), (
        f"{name}: nax_err={nax_err:.3e} tiled_err={tiled_err:.3e}"
    )
