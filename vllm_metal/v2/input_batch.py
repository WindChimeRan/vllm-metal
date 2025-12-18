# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible input batch functions using PyTorch instead of Triton kernels."""

import torch


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefill_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    """PyTorch implementation of prepare_prefill_inputs.

    Copies tokens from prefill_token_ids to input_ids based on query locations.
    """
    num_reqs = idx_mapping.shape[0]

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx].item())
        pf_len = int(prefill_len[req_state_idx].item())
        num_computed = int(num_computed_tokens[req_state_idx].item())

        if num_computed >= pf_len:
            # Not prefill
            continue

        query_start = int(query_start_loc[batch_idx].item())
        query_end = int(query_start_loc[batch_idx + 1].item())
        query_len = query_end - query_start

        # Copy tokens from prefill_token_ids to input_ids
        tokens = prefill_token_ids[
            req_state_idx, num_computed : num_computed + query_len
        ]
        input_ids[query_start:query_end] = tokens

        # Set next prefill token
        next_pos = num_computed + query_len
        if next_pos < pf_len:
            next_prefill_tokens[req_state_idx] = prefill_token_ids[
                req_state_idx, next_pos
            ]


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """PyTorch implementation of prepare_pos_seq_lens.

    Sets position indices and sequence lengths for each request.
    """
    num_reqs = idx_mapping.shape[0]
    max_num_reqs = seq_lens.shape[0]

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx].item())
        num_computed = int(num_computed_tokens[req_state_idx].item())

        query_start = int(query_start_loc[batch_idx].item())
        query_end = int(query_start_loc[batch_idx + 1].item())
        query_len = query_end - query_start

        # Set position indices
        for i in range(query_len):
            pos[query_start + i] = num_computed + i

        # Set sequence length
        seq_lens[batch_idx] = num_computed + query_len

    # Pad unused seq_lens as 0 for full CUDA graphs (matches Triton kernel behavior)
    seq_lens[num_reqs:max_num_reqs] = 0


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
) -> torch.Tensor:
    """PyTorch implementation of combine_sampled_and_draft_tokens.

    Combines sampled and draft tokens based on accepted lengths.
    Returns logits_indices tensor.
    """
    num_reqs = seq_lens.shape[0]

    logits_indices = torch.empty(
        num_logits,
        dtype=torch.int64,
        device=input_ids.device,
    )

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx].item())

        # Get the number of logits and draft tokens
        cu_num_logits_start = int(cu_num_logits[batch_idx].item())
        cu_num_logits_end = int(cu_num_logits[batch_idx + 1].item())
        num_logits_for_req = cu_num_logits_end - cu_num_logits_start
        num_draft_tokens = num_logits_for_req - 1

        # Compute the logits indices
        query_end = int(query_start_loc[batch_idx + 1].item())
        logits_start = query_end - num_logits_for_req
        for i in range(num_logits_for_req):
            logits_indices[cu_num_logits_start + i] = logits_start + i

        seq_len = int(seq_lens[batch_idx].item())
        pf_len = int(prefill_len[req_state_idx].item())
        if seq_len <= pf_len:
            # Handling prefill tokens. No sampled or draft tokens.
            continue

        # Write the last sampled token ID to input_ids
        last_token_id = int(last_sampled_tokens[req_state_idx].item())
        input_ids[query_end - num_logits_for_req] = last_token_id

        # Write the draft tokens (if any) to input_ids
        if num_draft_tokens > 0:
            for i in range(num_draft_tokens):
                draft_token = int(draft_tokens[req_state_idx, i].item())
                input_ids[query_end - num_draft_tokens + i] = draft_token

    return logits_indices


def post_update(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    """PyTorch implementation of post_update.

    Updates internal state after model execution.
    """
    num_reqs = idx_mapping.shape[0]

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx].item())

        num_sampled_for_req = int(num_sampled[batch_idx].item())
        if num_sampled_for_req > 0:
            # Store the last sampled token
            token_id = int(sampled_tokens[batch_idx, num_sampled_for_req - 1].item())
            last_sampled_tokens[req_state_idx] = token_id

        # Update output_bin_counts for all sampled tokens
        for i in range(num_sampled_for_req):
            token_id = int(sampled_tokens[batch_idx, i].item())
            output_bin_counts[req_state_idx, token_id] += 1

        # Update num_computed_tokens
        query_start = int(query_start_loc[batch_idx].item())
        query_end = int(query_start_loc[batch_idx + 1].item())
        query_len = query_end - query_start
        num_rejected_for_req = int(num_rejected[batch_idx].item())

        num_computed = int(num_computed_tokens[req_state_idx].item())
        num_computed += query_len - num_rejected_for_req
        num_computed_tokens[req_state_idx] = num_computed
