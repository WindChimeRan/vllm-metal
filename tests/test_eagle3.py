# SPDX-License-Identifier: Apache-2.0
"""EAGLE3 numerical reference and prefix/cache lifecycle regression tests."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from vllm import SamplingParams

from vllm_metal.v1.eagle3 import Eagle3Config, Eagle3Model
from vllm_metal.v1.eagle3_proposer import Eagle3Proposer
from vllm_metal.v1.model_runner import PrefillRequest
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController


def _model(kv_heads=2, norm_before_residual=True):
    target = {
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 128,
    }
    config = Eagle3Config.from_dict(
        {
            "speculators_model_type": "eagle3",
            "draft_vocab_size": 64,
            "norm_before_residual": norm_before_residual,
            "transformer_layer_config": {
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": kv_heads,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "vocab_size": 128,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
            },
        },
        target,
    )
    model = Eagle3Model(config)
    model.d2t = mx.arange(64, dtype=mx.int32)
    mx.eval(model.parameters())
    return model


def _reference(model, tokens, features):
    """Independent PyTorch implementation of the checkpoint's dense equations."""
    weights = {
        name: torch.from_numpy(np.array(value))
        for name, value in tree_flatten(model.parameters())
    }
    tokens = torch.from_numpy(np.array(tokens).astype(np.int64))
    hidden = torch.from_numpy(np.array(features))

    def linear(x, name):
        return torch_f.linear(x, weights[name + ".weight"])

    def norm(x, name):
        return torch_f.rms_norm(x, (x.shape[-1],), weights[name + ".weight"], 1e-6)

    hidden = linear(hidden, "fc")
    embeddings = torch_f.embedding(tokens, weights["embed_tokens.weight"])
    residual = (
        norm(hidden, "layers.0.hidden_norm")
        if model.config.norm_before_residual
        else hidden
    )
    x = torch.cat(
        (
            norm(embeddings, "layers.0.input_layernorm"),
            norm(hidden, "layers.0.hidden_norm"),
        ),
        -1,
    )
    b, length, _ = x.shape
    queries = (
        linear(x, "layers.0.self_attn.q_proj").view(b, length, 4, 16).transpose(1, 2)
    )
    kv_heads = model.config.layer.num_key_value_heads
    keys = (
        linear(x, "layers.0.self_attn.k_proj")
        .view(b, length, kv_heads, 16)
        .transpose(1, 2)
    )
    values = (
        linear(x, "layers.0.self_attn.v_proj")
        .view(b, length, kv_heads, 16)
        .transpose(1, 2)
    )
    angles = torch.arange(length)[:, None] * (
        10000.0 ** (-torch.arange(0, 16, 2).float() / 16)
    )
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]

    def rope(x):
        left, right = x.chunk(2, -1)
        return torch.cat(
            (left * cosine - right * sine, right * cosine + left * sine), -1
        )

    attention = torch_f.scaled_dot_product_attention(
        rope(queries), rope(keys), values, is_causal=True, enable_gqa=True
    )
    hidden = residual + linear(
        attention.transpose(1, 2).reshape(b, length, 64), "layers.0.self_attn.o_proj"
    )
    x = norm(hidden, "layers.0.post_attention_layernorm")
    hidden = hidden + linear(
        torch_f.silu(linear(x, "layers.0.mlp.gate_proj"))
        * linear(x, "layers.0.mlp.up_proj"),
        "layers.0.mlp.down_proj",
    )
    return norm(hidden, "norm").numpy(), hidden.numpy()


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("norm_before_residual", [True, False])
def test_head_matches_pytorch_gqa_mha_reference(kv_heads, norm_before_residual):
    model = _model(kv_heads, norm_before_residual)
    tokens = mx.array([[1, 9, 7, 22, 63], [8, 3, 12, 91, 24]])
    features = mx.random.normal((2, 5, 192))
    expected, expected_recurrent = _reference(model, tokens, features)
    # Check the equations in full FP32. MLX's M5 GPU matmul defaults to
    # TF32 (MLX_ENABLE_TF32), while this PyTorch reference uses CPU FP32.
    # The paged/ragged tests below exercise the GPU execution separately.
    with mx.stream(mx.cpu):
        actual, recurrent = model(
            tokens, model.combine_hidden_states(features), mask="causal"
        )
    np.testing.assert_allclose(np.array(actual), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        np.array(recurrent), expected_recurrent, rtol=2e-5, atol=2e-5
    )


def _proposer(model):
    proposer = Eagle3Proposer.build(
        model=model,
        controller=SpeculativeDecodeController(),
        committed_num_blocks=16,
        scratch_reserve_blocks=8,
        block_size=16,
        dtype=mx.float32,
    )
    proposer.adopt_committed_group(0)
    return proposer


def _prefill(proposer, requests, *, k=3, states=None, finished=()):
    """requests = (id, full prompt, feature rows, cache blocks, start, end)."""
    prefills, sampled, modes, feature_rows = [], [], [], []
    states = dict(states or {})
    cu = [0]
    for req_id, prompt, features, blocks, start, end in requests:
        params = SamplingParams(temperature=0)
        intermediate = end < len(prompt) - 1
        prefills.append(
            PrefillRequest(
                req_id=req_id,
                token_ids=prompt[start:end],
                sampling_params=params,
                block_ids=[blocks],
                generator=None,
                prompt_len=None if intermediate else end,
                start_pos=start,
                full_prompt_token_ids=prompt[:-1],
            )
        )
        sampled.append(-1 if intermediate else prompt[end])
        modes.append("intermediate" if intermediate else "new_final")
        feature_rows.append(features[start:end])
        cu.append(cu[-1] + end - start)
        states[req_id] = SimpleNamespace(token_ids=prompt, sampling_params=params)
    return proposer.propose(
        ProposeContext(
            target_hidden_states=mx.concatenate(feature_rows),
            decode_reqs=[],
            decode_segments=[],
            decode_token_ids=[],
            prefill_reqs=prefills,
            prefill_token_ids=sampled,
            prefill_result_modes=modes,
            request_states=states,
            cu_seqlens=cu,
            num_decode_segments=0,
            num_speculative_tokens=k,
            finished_req_ids=set(finished),
        )
    )


def _plain_drafts(model, prompt, features, k=3, cache=None):
    cache = KVCache() if cache is None else cache
    normalized, hidden = model(
        mx.array(prompt[1:])[None],
        model.combine_hidden_states(features)[None],
        mask="causal",
        cache=cache,
    )
    columns = [model.top_tokens(normalized[:, -1])]
    hidden = hidden[:, -1:]
    for _ in range(k - 1):
        normalized, hidden = model(columns[-1][:, None], hidden, cache=cache)
        columns.append(model.top_tokens(normalized[:, 0]))
    return mx.stack(columns, axis=1).tolist()[0]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_ragged_batch_matches_independent_full_context_drafts(k):
    model = _model()
    proposer = _proposer(model)
    prompts = [list(range(35)), list(range(21, 43))]
    features = [mx.random.normal((len(p) - 1, 192)) for p in prompts]
    result = _prefill(
        proposer,
        [
            ("a", prompts[0], features[0], [1, 2, 3], 0, 34),
            ("b", prompts[1], features[1], [4, 5], 0, 21),
        ],
        k=k,
    )
    assert result.draft_token_ids == [
        _plain_drafts(model, p, f, k) for p, f in zip(prompts, features, strict=False)
    ]


def test_prefix_hit_with_different_next_token_preserves_shared_kv():
    model = _model()
    proposer = _proposer(model)
    original = list(range(35))
    features = mx.random.normal((34, 192))
    _prefill(proposer, [("a", original, features, [1, 2, 3], 0, 34)])
    # Freeze values on the CPU; another MLX array can alias in-place writes.
    shared_keys = np.array(proposer._kv.key_caches[0][1]).copy()
    changed = original[:16] + [87, 92, 12, 41, 56, 22, 75]
    changed_features = mx.concatenate((features[:16], mx.random.normal((6, 192))))
    result = _prefill(
        proposer, [("b", changed, changed_features, [1, 4], 16, 22)], finished=("a",)
    )
    reference_cache = KVCache()
    assert result.draft_token_ids == [
        _plain_drafts(model, changed, changed_features, cache=reference_cache)
    ]
    np.testing.assert_array_equal(shared_keys, np.array(proposer._kv.key_caches[0][1]))
    physical = [16 + i for i in range(1, 16)] + [64 + i for i in range(7)]
    for pool, reference in (
        (proposer._kv.key_caches[0], reference_cache.keys),
        (proposer._kv.value_caches[0], reference_cache.values),
    ):
        actual = pool.reshape(-1, 2, 16)[mx.array(physical)].transpose(1, 0, 2)
        np.testing.assert_allclose(
            np.array(actual), np.array(reference[0, :, :22]), atol=3e-3, rtol=3e-3
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_prefix_created_in_the_same_batch_uses_its_current_boundary_feature(reverse):
    model = _model()
    proposer = _proposer(model)
    first = list(range(35))
    first_features = mx.random.normal((34, 192))
    second = first[:16] + [87, 92, 12, 41, 56, 22, 75]
    second_features = mx.concatenate((first_features[:16], mx.random.normal((6, 192))))
    requests = [
        ("owner", first, first_features, [1, 2, 3], 0, 34),
        ("hit", second, second_features, [1, 4], 16, 22),
    ]
    result = _prefill(proposer, requests[::-1] if reverse else requests)
    reference_cache = KVCache()
    expected = _plain_drafts(model, second, second_features, cache=reference_cache)
    # The first uncached KV row depends on the boundary feature produced by
    # the owner in THIS forward; the persistent sidecar was zero before it.
    for pool, reference in (
        (proposer._kv.key_caches[0], reference_cache.keys),
        (proposer._kv.value_caches[0], reference_cache.values),
    ):
        np.testing.assert_allclose(
            np.array(pool[4, 0]), np.array(reference[0, :, 15]), atol=3e-3, rtol=3e-3
        )
    assert result.draft_token_ids[0 if reverse else 1] == expected


def test_chunked_prefill_and_zero_budget_populate_reusable_prefix():
    model = _model()
    proposer = _proposer(model)
    prompt = list(range(51))
    features = mx.random.normal((50, 192))
    assert _prefill(proposer, [("a", prompt, features, [1], 0, 16)], k=0) is None
    assert _prefill(proposer, [("a", prompt, features, [1, 2], 16, 32)], k=0) is None
    result = _prefill(proposer, [("a", prompt, features, [1, 2, 3, 4], 32, 50)])
    expected = _plain_drafts(model, prompt, features)
    assert result.draft_token_ids == [expected]
    result = _prefill(
        proposer, [("b", prompt, features, [1, 2, 5, 6], 32, 50)], finished=("a",)
    )
    assert result.draft_token_ids == [expected]


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_verification_rebuilds_kv_from_actual_target_features(accepted):
    model = _model()
    proposer = _proposer(model)
    prompt = list(range(32))
    features = mx.random.normal((31, 192))
    previous = _prefill(proposer, [("a", prompt, features, [1, 2], 0, 31)])
    drafts = previous.draft_token_ids[0]
    output = drafts[:accepted] + [79]
    verified_features = mx.random.normal((4, 192))
    committed = prompt + output
    params = SamplingParams(temperature=0)
    state = SimpleNamespace(
        token_ids=committed, block_ids=[[1, 2, 3]], sampling_params=params
    )
    segment = PagedDecodeSegment(
        req_id="a",
        input_token_ids=(prompt[-1], *drafts),
        start_row=0,
        num_query_tokens=4,
        draft_token_ids=tuple(drafts),
        cache_start_pos=31,
        block_ids=((1, 2, 3),),
    )
    result = proposer.propose(
        ProposeContext(
            target_hidden_states=verified_features,
            decode_reqs=[("a", state)],
            decode_segments=[segment],
            decode_token_ids=[output],
            prefill_reqs=[],
            prefill_token_ids=[],
            prefill_result_modes=[],
            request_states={"a": state},
            cu_seqlens=[0, 4],
            num_decode_segments=1,
            num_speculative_tokens=3,
            finished_req_ids=set(),
        )
    )
    canonical = mx.concatenate((features, verified_features[: accepted + 1]))
    assert result.draft_token_ids == [_plain_drafts(model, committed, canonical)]


def test_finished_id_can_be_reused_without_retaining_old_draft_state():
    model = _model()
    proposer = _proposer(model)
    long_prompt = list(range(32))
    _prefill(
        proposer, [("same-id", long_prompt, mx.random.normal((31, 192)), [1, 2], 0, 31)]
    )
    prompt = [12, 48, 7, 91, 36, 52, 67, 11]
    features = mx.random.normal((7, 192))
    result = _prefill(
        proposer, [("same-id", prompt, features, [1], 0, 7)], finished=("same-id",)
    )
    assert result.draft_token_ids == [_plain_drafts(model, prompt, features)]
    assert not proposer._scratch_req_blocks


def test_quantized_checkpoint_preserves_packed_weights_and_dense_embeddings(tmp_path):
    import mlx.nn as nn

    model = _model()
    nn.quantize(
        model,
        bits=4,
        group_size=32,
        class_predicate=lambda _, module: isinstance(module, nn.Linear),
    )
    mx.eval(model.parameters())
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    config = {
        "speculators_model_type": "eagle3",
        "draft_vocab_size": 64,
        "norm_before_residual": True,
        "transformer_layer_config": asdict(model.config.layer),
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = Eagle3Model.load(
        str(tmp_path),
        {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 8,
            "vocab_size": 128,
        },
        mx.float32,
    )
    assert loaded.fc.weight.dtype == mx.uint32
    assert loaded.embed_tokens.weight.dtype == mx.float32
    tokens = mx.array([[1, 2, 3]])
    features = mx.random.normal((1, 3, 192))
    expected, _ = model(tokens, model.combine_hidden_states(features), mask="causal")
    actual, _ = loaded(tokens, loaded.combine_hidden_states(features), mask="causal")
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_original_checkpoint_loads_midlayer_weights_and_borrows_target_embedding(
    tmp_path,
):
    model = _model(norm_before_residual=False)
    config = {**asdict(model.config.layer), "draft_vocab_size": 64}
    (tmp_path / "config.json").write_text(json.dumps(config))
    state = {
        name.replace("layers.0.", "midlayer."): torch.from_numpy(np.array(value))
        for name, value in tree_flatten(model.parameters())
        if name != "embed_tokens.weight"
    }
    torch.save(state, tmp_path / "pytorch_model.bin")
    target = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 128,
    }
    with pytest.raises(ValueError, match="target's dense embedding"):
        Eagle3Model.load(str(tmp_path), target, mx.float32)
    loaded = Eagle3Model.load(
        str(tmp_path), target, mx.float32, target_embedding=model.embed_tokens
    )
    assert loaded.config.norm_before_residual is False
    assert loaded.embed_tokens is not model.embed_tokens
    tokens, features = mx.array([[1, 2, 3]]), mx.random.normal((1, 3, 192))
    expected, _ = model(tokens, model.combine_hidden_states(features), mask="causal")
    actual, _ = loaded(tokens, loaded.combine_hidden_states(features), mask="causal")
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_preempted_request_replays_output_history_beyond_original_prompt():
    model = _model()
    proposer = _proposer(model)
    history = list(range(41))
    features = mx.random.normal((40, 192))
    _prefill(proposer, [("r", history[:17], features[:16], [1], 0, 16)], k=0)
    proposer.release_requests({"r"})
    state = SimpleNamespace(
        token_ids=history, sampling_params=SamplingParams(temperature=0)
    )
    prefill = PrefillRequest(
        req_id="r",
        token_ids=history[16:32],
        sampling_params=state.sampling_params,
        block_ids=[[1, 2]],
        generator=None,
        prompt_len=None,
        start_pos=16,
        full_prompt_token_ids=history[:16],
    )
    result = proposer.propose(
        ProposeContext(
            target_hidden_states=features[16:32],
            decode_reqs=[],
            decode_segments=[],
            decode_token_ids=[],
            prefill_reqs=[prefill],
            prefill_token_ids=[-1],
            prefill_result_modes=["intermediate"],
            request_states={"r": state},
            cu_seqlens=[0, 16],
            num_decode_segments=0,
            num_speculative_tokens=0,
            finished_req_ids=set(),
        )
    )
    assert result is None
    final = _prefill(proposer, [("r", history, features, [1, 2, 3], 32, 40)])
    assert final.draft_token_ids == [_plain_drafts(model, history, features)]


def test_trace_keeps_only_emitted_verification_rows():
    from types import SimpleNamespace

    import mlx.core as mx
    from vllm import SamplingParams

    from tools.benchmark.eagle3_benchmark import metal_trace
    from vllm_metal.v1.sampling_batch import SamplingBatch
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

    controller = SpeculativeDecodeController()
    runner = SimpleNamespace(_spec_decode_controller=controller)
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(model_runner=runner)
            )
        ),
        get_tokenizer=lambda: SimpleNamespace(decode=str),
    )
    params = SamplingParams(temperature=0)
    # First request accepts both drafts and emits a bonus; the second
    # rejects immediately, so its later verification rows are not outputs.
    segments = [
        PagedDecodeSegment("a", (0, 1, 2), 0, 3, (1, 2), 1, ((0,),)),
        PagedDecodeSegment("b", (0, 2, 1), 3, 3, (2, 1), 1, ((1,),)),
    ]
    requests = [
        (
            name,
            SimpleNamespace(
                token_ids=[prompt, 0], prompt_len=1, sampling_params=params
            ),
        )
        for name, prompt in (("a", 10), ("b", 11))
    ]
    logits = mx.array(
        [
            [
                [0.0, 3.0, 1.0, 2.0],
                [0.0, 1.0, 3.0, 2.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 3.0, 1.0, 2.0],
                [0.0, 1.0, 3.0, 2.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        ]
    )
    original = controller.verify_greedy
    with metal_trace(llm, 2) as (scores, counts):
        import vllm_metal.v1.sampling_batch as sampling

        batch = SamplingBatch([params], [[10]], [[]], vocab_size=4)
        first = sampling.sample_from_logits(
            mx.array([[3.0, 2.0, 1.0, 0.0]]), batch, None
        )
        assert first.token_ids == [0]
        assert params.logprobs is None
        assert controller.verify_greedy(logits, requests, segments) == [
            [1, 2, 3],
            [1],
        ]
    assert controller.verify_greedy == original
    assert counts == {"rounds": 2, "drafted": 4, "accepted": 2}
    assert scores[(10,), ()][0]["id"] == 0
    assert scores[(10,), (0, 1, 2)][0]["id"] == 3
    assert scores[(11,), (0,)][0]["id"] == 1
    assert ((11,), (0, 1)) not in scores
