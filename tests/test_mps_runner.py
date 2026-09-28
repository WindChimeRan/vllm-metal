# SPDX-License-Identifier: Apache-2.0
"""Packed inputs and persistent state updates for the upstream MRV2 runner."""

import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.states import RequestState

from vllm_metal.pytorch_backend import input_ops
from vllm_metal.pytorch_backend.runtime import install

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS required"
)


@pytest.fixture
def state():
    install()
    return RequestState(4, 64, 16, 0, 64, torch.device("mps"))


def test_upstream_output_copy_waits_for_mps_results():
    install()
    sampled = (torch.arange(3, device="mps") + 40).unsqueeze(1)
    output = AsyncOutput(
        ModelRunnerOutput(
            req_ids=["a", "b", "c"], req_id_to_index={"a": 0, "b": 1, "c": 2}
        ),
        SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=None,
        ),
        torch.tensor([1, 0, 1], device="mps", dtype=torch.int32),
        torch.cuda.current_stream(torch.device("mps")),
        torch.cuda.Stream(torch.device("mps")),
    )
    assert output.get_output().sampled_token_ids == [[40], [], [42]]


def test_resumption_keeps_prompt_length_and_generated_history(state):
    state.add_request("r", 3, [1, 2, 3], 0, 20)
    state.apply_staged_writes()
    state.remove_request("r")
    state.add_request("r", 3, [1, 2, 3, 9, 10], 3, 20)
    state.apply_staged_writes()
    slot = state.req_id_to_index["r"]
    assert state.prompt_len.gpu[slot].item() == 3
    assert state.prefill_len.gpu[slot].item() == 5
    assert state.all_token_ids.gpu[slot, :5].tolist() == [1, 2, 3, 9, 10]
    assert state.num_computed_tokens.gpu[slot].item() == 3


def test_mixed_prefill_decode_and_writeback(state):
    state.add_request("prefill", 6, [10, 11, 12, 13, 14, 15], 2, 20)
    state.add_request("decode", 3, [30, 31, 32, 40], 4, 20)
    state.apply_staged_writes()
    p, d = [state.req_id_to_index[k] for k in ("prefill", "decode")]
    # Last sampled token is present in history but has not been computed yet.
    state.num_computed_tokens.gpu[d] = 3
    state.prefill_len.np[d] = 3
    state.prefill_len.copy_to_uva()
    state.last_sampled_tokens[d, 0] = 40
    mapping = torch.tensor([d, p], dtype=torch.int32, device="mps")
    cu = torch.tensor([0, 1, 4], dtype=torch.int32, device="mps")
    ids = torch.zeros(8, dtype=torch.int32, device="mps")
    pos = torch.zeros(8, dtype=torch.int64, device="mps")
    lengths = torch.zeros(4, dtype=torch.int32, device="mps")
    input_ops.prepare_prefill_inputs(
        ids,
        state.next_prefill_tokens,
        mapping,
        cu,
        state.all_token_ids.gpu,
        state.prefill_len.gpu,
        state.num_computed_tokens.gpu,
    )
    input_ops.prepare_pos_seq_lens(
        mapping, cu, state.num_computed_tokens.gpu, pos, lengths
    )
    cu_logits = torch.tensor([0, 1, 2], dtype=torch.int32, device="mps")
    indices = input_ops.combine_sampled_and_draft_tokens(
        ids,
        mapping,
        state.last_sampled_tokens,
        cu,
        lengths,
        state.prefill_len.gpu,
        state.draft_tokens,
        cu_logits,
        2,
    )
    assert ids[:4].tolist() == [40, 12, 13, 14]
    assert pos[:4].tolist() == [3, 2, 3, 4]
    assert lengths.tolist() == [4, 5, 0, 0]
    assert indices.tolist() == [0, 3]
    counts, rejected = input_ops.get_num_sampled_and_rejected(
        torch.ones(2, dtype=torch.int32, device="mps"),
        lengths,
        cu_logits,
        mapping,
        state.prefill_len.gpu,
    )
    assert counts.tolist() == [1, 0]
    input_ops.post_update(
        mapping,
        state.num_computed_tokens.gpu,
        state.last_sampled_tokens,
        None,
        torch.tensor([[41], [16]], device="mps"),
        counts,
        rejected,
        cu,
        state.all_token_ids.gpu,
        state.total_len.gpu,
    )
    assert state.all_token_ids.gpu[d, :5].tolist() == [30, 31, 32, 40, 41]
    assert state.total_len.gpu[[d, p]].tolist() == [5, 6]
    assert state.num_computed_tokens.gpu[[d, p]].tolist() == [4, 5]


def test_block_mapping_uses_request_slots_and_padding(state):
    tables = BlockTables([16], 4, 16, [4], torch.device("mps"), [16])
    tables.append_block_ids(3, ([8, 9],), True)
    tables.append_block_ids(1, ([2, 4],), True)
    tables.apply_staged_writes()
    mapping = torch.tensor([1, 3], dtype=torch.int32, device="mps")
    (gathered,) = tables.gather_block_tables(mapping, num_reqs_padded=3)
    assert gathered[:, :2].tolist() == [[2, 4], [8, 9], [0, 0]]
    cu = torch.tensor([0, 1, 4], dtype=torch.int32, device="mps")
    positions = torch.tensor([17, 14, 15, 16], dtype=torch.int64, device="mps")
    slots = tables.compute_slot_mappings(mapping, cu, positions, num_tokens_padded=8)
    assert slots.tolist() == [[65, 142, 143, 144, -1, -1, -1, -1]]


def test_writeback_skips_removed_request_without_overwriting_slot_zero(state):
    state.num_computed_tokens.gpu[0] = 3
    state.total_len.gpu[0] = 4
    state.last_sampled_tokens[0, 0] = 40
    mapping = torch.tensor([0, -1], dtype=torch.int32, device="mps")
    input_ops.post_update(
        mapping,
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
    assert state.all_token_ids.gpu[0, 4].item() == 41
    assert state.total_len.gpu[0].item() == 5
    assert state.num_computed_tokens.gpu[0].item() == 4
