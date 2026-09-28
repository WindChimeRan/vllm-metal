# SPDX-License-Identifier: Apache-2.0
"""MPS-specific request mapping, sampling and RNG contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_metal.pytorch_backend.input_ops import post_update
from vllm_metal.pytorch_backend.runtime import install
from vllm_metal.pytorch_backend.sampling_ops import gumbel_sample

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


@pytest.fixture
def state():
    install()
    return RequestState(4, 64, 16, 0, 64, torch.device("mps"))


def test_request_slots_and_removed_request_writeback(state):
    tables = BlockTables([16], 4, 16, [4], torch.device("mps"), [16])
    tables.append_block_ids(3, ([8, 9],), True)
    tables.append_block_ids(0, ([2, 4],), True)
    tables.apply_staged_writes()
    mapping = torch.tensor([0, 3], dtype=torch.int32, device="mps")
    (gathered,) = tables.gather_block_tables(mapping, num_reqs_padded=3)
    assert gathered[:, :2].tolist() == [[2, 4], [8, 9], [0, 0]]
    slots = tables.compute_slot_mappings(
        mapping,
        torch.tensor([0, 1, 4], dtype=torch.int32, device="mps"),
        torch.tensor([17, 14, 15, 16], dtype=torch.int64, device="mps"),
        num_tokens_padded=8,
    )
    assert slots.tolist() == [[65, 142, 143, 144, -1, -1, -1, -1]]
    # The second request disappears before writeback. Its -1 slot must not
    # alias slot zero and overwrite the surviving request's sampled token.
    state.num_computed_tokens.gpu[0] = 17
    state.total_len.gpu[0] = 18
    state.last_sampled_tokens[0, 0] = 40
    post_update(
        torch.tensor([0, -1], dtype=torch.int32, device="mps"),
        state.num_computed_tokens.gpu,
        state.last_sampled_tokens,
        None,
        torch.tensor([[41], [999]], device="mps"),
        torch.ones(2, dtype=torch.int32, device="mps"),
        torch.zeros(2, dtype=torch.int32, device="mps"),
        torch.tensor([0, 1, 2], dtype=torch.int32, device="mps"),
        state.all_token_ids.gpu,
        state.total_len.gpu,
    )
    assert state.last_sampled_tokens[0, 0].item() == 41
    assert state.all_token_ids.gpu[0, 18].item() == 41
    assert state.total_len.gpu[0].item() == 19
    assert state.num_computed_tokens.gpu[0].item() == 18


def test_upstream_sampler_controls_and_mixed_logprobs(state):
    sampler = Sampler(4, 64, torch.device("mps"), state)
    params = [
        SamplingParams(
            temperature=0,
            presence_penalty=0.3,
            frequency_penalty=0.4,
            repetition_penalty=1.2,
            logit_bias={31: 8},
            min_tokens=4,
            stop_token_ids=[7],
            allowed_token_ids=[7, 8, 10, 31],
            logprobs=3,
        ),
        SamplingParams(temperature=0, logprob_token_ids=[2, 5]),
    ]
    slots = []
    for i, p in enumerate(params):
        state.add_request(str(i), 3, [1, 2, 3, 10, 10], 4, 16)
        slot = state.req_id_to_index[str(i)]
        slots.append(slot)
        sampler.add_request(slot, 3, p)
    state.apply_staged_writes()
    sampler.apply_staged_writes()
    mapping = torch.tensor(slots, dtype=torch.int32, device="mps")
    batch = SimpleNamespace(
        expanded_idx_mapping=mapping,
        idx_mapping=mapping,
        idx_mapping_np=np.array(slots, dtype=np.int32),
        cu_num_logits_np=np.arange(3, dtype=np.int32),
        cu_num_logits=torch.arange(3, dtype=torch.int32, device="mps"),
        expanded_local_pos=torch.zeros(2, dtype=torch.int32, device="mps"),
        positions=torch.full((2,), 4, device="mps"),
        logits_indices=torch.arange(2, device="mps"),
        input_ids=torch.full((2,), 10, device="mps"),
        seq_lens=torch.full((2,), 5, dtype=torch.int32, device="mps"),
        num_reqs=2,
    )
    logits = torch.linspace(-1, 1, 64, device="mps").repeat(2, 1)
    result = sampler(logits.clone(), batch)
    assert result.sampled_token_ids.tolist() == [[31], [63]]
    values = result.logprobs_tensors
    expected = logits.log_softmax(-1).gather(1, values.logprob_token_ids.long())
    expected[1, -1] = -torch.inf  # Padding after two custom logprob tokens.
    torch.testing.assert_close(values.logprobs, expected, atol=2e-6, rtol=1e-5)
    assert values.logprob_token_ids[1, 1:3].tolist() == [2, 5]
    assert values.selected_token_ranks.tolist() == [33, 1]
    processed = sampler.apply_sampling_params(
        logits.clone(),
        mapping,
        mapping,
        batch.idx_mapping_np,
        batch.positions,
        batch.input_ids,
        batch.expanded_local_pos,
    )
    assert processed[0, 7].isneginf()  # min_tokens suppresses stop tokens.
    assert processed[0, 0].isneginf()  # Outside the allowlist.
    torch.testing.assert_close(processed[0, 10], logits[0, 10] * 1.2 - 0.4 * 2 - 0.3)


def test_seeded_sampling_is_independent_of_batch_order():
    logits = torch.randn(3, 64, device="mps")
    temperature = torch.tensor([0.7, 0, 0.8], device="mps")
    seeds = torch.tensor([42, 17, 23], device="mps")
    mapping = torch.arange(3, device="mps")
    positions = torch.tensor([9, 12, 15], device="mps")
    first = gumbel_sample(logits, mapping, temperature, seeds, positions, True, False)
    order = torch.tensor([2, 0, 1], device="mps")
    second = gumbel_sample(
        logits[order], mapping[order], temperature, seeds, positions[order], True, False
    )
    assert second.tolist() == first[order].tolist()
    assert first[1].item() == logits[1].argmax().item()


@pytest.mark.parametrize("model_type", ["hunyuan_v1_dense", "olmo2"])
def test_upstream_qk_norm_semantics(model_type, tmp_path, monkeypatch):
    from transformers import AutoConfig, AutoModelForCausalLM
    from vllm import LLM, envs

    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_METAL_BACKEND", "mps")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    torch.manual_seed(17)
    config = AutoConfig.for_model(
        model_type,
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        **({"head_dim": 128} if model_type == "hunyuan_v1_dense" else {}),
    )
    reference = AutoModelForCausalLM.from_config(
        config, dtype=torch.float16, attn_implementation="eager"
    ).eval()
    norm_count = 0
    with torch.no_grad():
        for name, weight in reference.named_parameters():
            if name.endswith(
                (
                    "query_layernorm.weight",
                    "key_layernorm.weight",
                    "q_norm.weight",
                    "k_norm.weight",
                )
            ):
                norm_count += 1
                # Uniform scales can hide an incorrect norm/RoPE order.
                weight.copy_(torch.linspace(0.5, 1.75, weight.numel()).view_as(weight))
    assert norm_count == 4
    reference.save_pretrained(tmp_path)
    prompt = [3 + i % 29 for i in range(35)]
    expected_ids, expected_scores = [], []
    with torch.inference_mode():
        for _ in range(3):
            logits = (
                reference(torch.tensor([prompt + expected_ids]), use_cache=False)
                .logits[0, -1]
                .float()
            )
            expected_ids.append(logits.argmax().item())
            expected_scores.append(logits.log_softmax(-1))
    del reference
    llm = LLM(
        model=str(tmp_path),
        model_impl="transformers",
        dtype="float16",
        skip_tokenizer_init=True,
        max_model_len=64,
        max_num_batched_tokens=16,
        max_num_seqs=2,
        kv_cache_memory_bytes=8 << 20,
        max_logprobs=64,
        enable_prefix_caching=True,
    )
    try:
        params = SamplingParams(
            temperature=0, max_tokens=3, ignore_eos=True, logprobs=-1
        )
        for cached_tokens in (0, 32):  # Cold prefill, then two cached blocks.
            result = llm.generate(
                [{"prompt_token_ids": prompt}], params, use_tqdm=False
            )[0]
            assert result.num_cached_tokens == cached_tokens
            output = result.outputs[0]
            assert output.token_ids == expected_ids
            for actual, expected in zip(output.logprobs, expected_scores, strict=True):
                scores = torch.tensor([actual[i].logprob for i in range(64)])
                torch.testing.assert_close(scores, expected, atol=0.01, rtol=0.002)
    finally:
        llm.llm_engine.engine_core.shutdown()
        envs.disable_envs_cache()
