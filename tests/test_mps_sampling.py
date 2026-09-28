# SPDX-License-Identifier: Apache-2.0
"""MRV2 sampling on MPS: controls, randomness and logprob contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_metal.pytorch_backend.runtime import install
from vllm_metal.pytorch_backend.sampling_ops import gumbel_sample

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


def setup(params, outputs=None):
    install()
    n = len(params)
    state = RequestState(n, 32, n, 0, 64, torch.device("mps"))
    sampler = Sampler(n, 64, torch.device("mps"), state)
    slots, positions, last = [], [], []
    for i, p in enumerate(params):
        tokens = [1, 2, 3] + (outputs[i] if outputs else [10, 10])
        state.add_request(str(i), 3, tokens, len(tokens) - 1, 16)
        slot = state.req_id_to_index[str(i)]
        slots.append(slot)
        positions.append(len(tokens) - 1)
        last.append(tokens[-1])
        sampler.add_request(slot, 3, p)
    state.apply_staged_writes()
    sampler.apply_staged_writes()
    mapping = torch.tensor(slots, dtype=torch.int32, device="mps")
    batch = SimpleNamespace(
        expanded_idx_mapping=mapping,
        idx_mapping=mapping,
        idx_mapping_np=np.array(slots, dtype=np.int32),
        cu_num_logits_np=np.arange(n + 1, dtype=np.int32),
        cu_num_logits=torch.arange(n + 1, dtype=torch.int32, device="mps"),
        expanded_local_pos=torch.zeros(n, dtype=torch.int32, device="mps"),
        positions=torch.tensor(positions, device="mps"),
        logits_indices=torch.arange(n, device="mps"),
        input_ids=torch.tensor(last, device="mps"),
        seq_lens=torch.tensor(
            [x + 1 for x in positions], dtype=torch.int32, device="mps"
        ),
        num_reqs=n,
    )
    return sampler, batch


def test_controls_and_logprobs():
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
        )
    ]
    sampler, batch = setup(params)
    logits = torch.linspace(-1, 1, 64, device="mps").unsqueeze(0)
    result = sampler(logits.clone(), batch)
    assert result.sampled_token_ids.item() == 31
    values = result.logprobs_tensors
    expected = logits.float().log_softmax(-1).gather(1, values.logprob_token_ids.long())
    torch.testing.assert_close(values.logprobs, expected, atol=2e-6, rtol=1e-5)
    assert values.selected_token_ranks.tolist() == [33]
    # Check the upstream processing path independently of the sampled argmax.
    processed = sampler.apply_sampling_params(
        logits.clone(),
        batch.expanded_idx_mapping,
        batch.idx_mapping,
        batch.idx_mapping_np,
        batch.positions,
        batch.input_ids,
        batch.expanded_local_pos,
    )
    assert processed[0, 7].isneginf()
    assert processed[0, 0].isneginf()
    expected_ten = logits[0, 10] * 1.2 - 0.4 * 2 - 0.3
    torch.testing.assert_close(processed[0, 10], expected_ten)


def test_custom_and_topk_logprobs():
    sampler, batch = setup(
        [
            SamplingParams(temperature=0, logprob_token_ids=[2, 5]),
            SamplingParams(temperature=0, logprobs=3),
        ]
    )
    logits = torch.randn(2, 64, device="mps")
    result = sampler(logits.clone(), batch).logprobs_tensors
    assert result.logprob_token_ids[0, 1:3].tolist() == [2, 5]
    assert result.logprobs[0, 3].isneginf()
    torch.testing.assert_close(
        result.logprobs[0, 1:3], logits.log_softmax(-1)[0, [2, 5]]
    )
    assert (
        result.logprob_token_ids[1, 1:].tolist() == logits[1].topk(3).indices.tolist()
    )


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


def test_zero_exponential_noise_cannot_select_excluded_token(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "exponential_", lambda self, **kw: self.zero_())
    logits = torch.full((4, 64), -torch.inf, device="mps")
    logits[:, 7] = 0
    args = [
        torch.arange(4, device="mps"),
        torch.ones(4, device="mps"),
        torch.arange(4, device="mps"),
        torch.zeros(4, dtype=torch.int64, device="mps"),
    ]
    assert gumbel_sample(logits, *args, False, False).tolist() == [7] * 4


def test_bad_words_match_output_suffix_only():
    params = SamplingParams(temperature=0)
    params._bad_words_token_ids = [[10, 11], [3, 12]]
    sampler, batch = setup([params], outputs=[[10]])
    logits = torch.zeros(1, 64, device="mps")
    logits[0, 11] = 10
    logits[0, 12] = 9
    result = sampler(logits, batch)
    assert result.sampled_token_ids.item() == 12
