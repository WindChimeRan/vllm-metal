# SPDX-License-Identifier: Apache-2.0
"""Numeric regression coverage for Hunyuan's post-RoPE Q/K norms."""

import mlx.core as mx
from mlx_lm.models.hunyuan_v1_dense import Attention, ModelArgs

from vllm_metal.attention.attention_contracts import attention_contract_for
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import prepare_sdpa_qkv


def _context(seq_len: int) -> PagedAttentionContext:
    return PagedAttentionContext(
        slot_mapping=list(range(seq_len)),
        block_tables=[[0]],
        context_lens=[seq_len],
        offsets=[],
        cu_seqlens=[0, seq_len],
    )


def test_hunyuan_qk_norm_is_applied_after_rope() -> None:
    args = ModelArgs(
        model_type="hunyuan_v1_dense",
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-5,
        head_dim=4,
    )
    attention = Attention(args)

    # Non-uniform learned norm weights make the two orders observably different.
    attention.query_layernorm.weight = mx.array(
        [0.5, 0.75, 1.25, 1.75], dtype=mx.bfloat16
    )
    attention.key_layernorm.weight = mx.array(
        [1.5, 0.625, 1.125, 0.875], dtype=mx.bfloat16
    )
    attention.q_proj.weight = (
        mx.arange(64).reshape(8, 8).astype(mx.float32) / 31.0 - 1.0
    ).astype(mx.bfloat16)
    attention.k_proj.weight = (
        mx.arange(32).reshape(4, 8).astype(mx.float32) / 19.0 - 0.75
    ).astype(mx.bfloat16)
    attention.v_proj.weight = mx.ones((4, 8), dtype=mx.bfloat16)

    x = (mx.arange(24).reshape(1, 3, 8).astype(mx.float32) / 11.0 - 1.0).astype(
        mx.bfloat16
    )
    raw_q = attention.q_proj(x).reshape(1, 3, 2, 4).transpose(0, 2, 1, 3)
    raw_k = attention.k_proj(x).reshape(1, 3, 1, 4).transpose(0, 2, 1, 3)

    expected_q = attention.query_layernorm(attention.rope(raw_q))
    expected_k = attention.key_layernorm(attention.rope(raw_k))
    wrong_q = attention.rope(attention.query_layernorm(raw_q))
    wrong_k = attention.rope(attention.key_layernorm(raw_k))

    queries, keys, _, _, _ = prepare_sdpa_qkv(
        attention,
        x,
        _context(seq_len=3),
        attention.n_heads,
        attention.n_kv_heads,
        attention_contract=attention_contract_for(attention),
    )
    mx.eval(queries, keys, expected_q, expected_k, wrong_q, wrong_k)

    assert not mx.allclose(wrong_q, expected_q, rtol=0.0, atol=1e-3).item()
    assert not mx.allclose(wrong_k, expected_k, rtol=0.0, atol=1e-3).item()
    assert mx.array_equal(queries, expected_q).item()
    assert mx.array_equal(keys, expected_k).item()
