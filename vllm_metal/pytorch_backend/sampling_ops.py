# SPDX-License-Identifier: Apache-2.0
"""PyTorch fallbacks for MRV2's Triton-only sampling operations.

The Sampler, request state, processor order and top-k/top-p stay in vLLM.
MRV2's other device operations have no MPS dispatch in vLLM 0.30, so their
packed-state interfaces are adapted here. Random selection and repetition
penalties use shared vLLM PyTorch helpers; request/position RNG stays local.
"""

from types import SimpleNamespace

import torch
from vllm._custom_ops import apply_repetition_penalties
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.sample.ops.topk_topp_sampler import sample_with_exponential_noise

from vllm_metal.pytorch_backend.runtime import TensorKernel, cpu_mirror


def apply_temperature(logits, expanded_idx_mapping, temperature):
    temp = temperature[expanded_idx_mapping.long()]
    logits.div_(torch.where(temp == 0, 1, temp).unsqueeze(1))


def apply_min_p(logits, expanded_idx_mapping, min_p):
    threshold = min_p[expanded_idx_mapping.long()].log().unsqueeze(1)
    logits.masked_fill_(
        logits < logits.amax(dim=-1, keepdim=True) + threshold, -torch.inf
    )


def gumbel_sample(
    logits,
    expanded_idx_mapping,
    temperature,
    seed,
    pos,
    apply_temperature,
    is_drafting,
    logits_cache=None,
    logits_cache_col=None,
    use_fp64=False,
):
    if is_drafting or logits_cache is not None or use_fp64:
        raise NotImplementedError("MPS draft sampling and FP64 noise are unsupported")
    mapping = cpu_mirror(expanded_idx_mapping).long()
    temps = cpu_mirror(temperature)[mapping]
    if bool((temps == 0).all()):
        return logits.argmax(dim=-1)
    seeds = cpu_mirror(seed)[mapping].tolist()
    positions = pos.cpu().tolist()
    noise = torch.empty_like(logits, dtype=torch.float32)
    for row, (request_seed, position) in enumerate(zip(seeds, positions, strict=True)):
        # Stateless per request/position: batching and preemption do not advance
        # a shared generator. MPS has its own reproducible RNG stream.
        key = (request_seed + position * 0x9E3779B97F4A7C15) & ((1 << 63) - 1)
        generator = torch.Generator(device="mps").manual_seed(key)
        noise[row].exponential_(generator=generator)
    noise.clamp_min_(torch.finfo(torch.float32).tiny)
    scores = logits.float()
    temp = temperature[expanded_idx_mapping.long()]
    if apply_temperature:
        scores = scores / torch.where(temp == 0, 1, temp).unsqueeze(1)
    sampled = sample_with_exponential_noise(scores.softmax(-1), noise)
    if bool((temps == 0).any()):
        sampled = torch.where(temp == 0, logits.argmax(dim=-1), sampled)
    return sampled


def apply_logit_bias(
    logits,
    expanded_idx_mapping,
    pos,
    num_allowed_token_ids,
    allowed_token_ids,
    num_logit_bias,
    logit_bias_token_ids,
    logit_bias,
    min_lens,
    num_stop_token_ids,
    restore_when_all_masked,
    stop_token_ids,
    check_all_masked_rows=False,
):
    slots = expanded_idx_mapping.long()
    n, vocab = logits.shape
    allowed_count = num_allowed_token_ids[slots]
    cols = torch.arange(allowed_token_ids.shape[1], device=logits.device)
    mask = torch.zeros((n, vocab), device=logits.device, dtype=torch.int32)
    mask.scatter_add_(
        1, allowed_token_ids[slots].long(), (cols < allowed_count[:, None]).int()
    )
    logits.masked_fill_((allowed_count[:, None] > 0) & (mask == 0), -torch.inf)

    cols = torch.arange(logit_bias.shape[1], device=logits.device)
    values = torch.where(cols < num_logit_bias[slots, None], logit_bias[slots], 0)
    logits.scatter_add_(1, logit_bias_token_ids[slots].long(), values)

    cols = torch.arange(stop_token_ids.shape[1], device=logits.device)
    stop_mask = torch.zeros_like(mask)
    stop_mask.scatter_add_(
        1, stop_token_ids[slots].long(), (cols < num_stop_token_ids[slots, None]).int()
    )
    old = logits.clone() if check_all_masked_rows else None
    logits.masked_fill_(
        (pos[:, None] + 1 < min_lens[slots, None]) & (stop_mask > 0), -torch.inf
    )
    if old is not None:
        restore = (restore_when_all_masked[slots] != 0) & logits.amax(-1).isneginf()
        logits.copy_(torch.where(restore[:, None] & old.isfinite(), old, logits))


def bincount(
    mapping,
    all_ids,
    prompt_len,
    prefill_len,
    prompt_mask,
    output_counts,
    max_prefill_len,
):
    slots = mapping.long()
    tokens = all_ids[slots, :max_prefill_len].long()
    cols = torch.arange(max_prefill_len, device=tokens.device)[None, :]
    counts = torch.zeros(
        (slots.numel(), output_counts.shape[1]), dtype=torch.int32, device=tokens.device
    )
    counts.scatter_add_(1, tokens, (cols < prompt_len[slots, None]).int())
    # Pack the vocabulary membership bits in the upstream representation.
    padding = prompt_mask.shape[1] * 32 - counts.shape[1]
    bits = torch.nn.functional.pad(counts > 0, (0, padding)).view(slots.numel(), -1, 32)
    shifts = torch.arange(32, device=tokens.device, dtype=torch.int64)
    prompt_mask[slots] = (bits.long() << shifts).sum(-1).to(torch.int32)
    counts.zero_()
    counts.scatter_add_(
        1,
        tokens,
        ((cols >= prompt_len[slots, None]) & (cols < prefill_len[slots, None])).int(),
    )
    output_counts[slots] = counts


def apply_penalties(
    logits,
    expanded_idx_mapping,
    token_ids,
    expanded_local_pos,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    prompt_bin_mask,
    output_bin_counts,
):
    slots = expanded_idx_mapping.long()
    vocab = torch.arange(logits.shape[1], device=logits.device)
    prompt_seen = (
        (prompt_bin_mask[slots[:, None], vocab // 32] >> (vocab % 32)) & 1
    ).bool()
    counts = output_bin_counts[slots]
    apply_repetition_penalties(
        logits, prompt_seen, counts > 0, repetition_penalty[slots]
    )
    logits.sub_(counts * frequency_penalty[slots, None])
    logits.sub_((counts > 0) * presence_penalty[slots, None])


def apply_bad_words(
    logits,
    expanded_idx_mapping,
    bad_word_token_ids,
    bad_word_offsets,
    num_bad_words,
    all_token_ids,
    prompt_len,
    total_len,
    input_ids,
    expanded_local_pos,
    max_num_bad_words,
):
    mapping = cpu_mirror(expanded_idx_mapping).long().tolist()
    # Constraint lists are host-origin metadata; token history remains on MPS.
    words = bad_word_token_ids.cpu()
    offsets = bad_word_offsets.cpu()
    sizes = cpu_mirror(num_bad_words)
    for row, slot in enumerate(mapping):
        for word in range(int(sizes[slot])):
            start, end = offsets[slot, word : word + 2].tolist()
            prefix = words[slot, start : end - 1].to(logits.device)
            count = end - start - 1
            positions = (
                total_len[slot].long()
                - count
                + torch.arange(count, device=logits.device)
            )
            matches = (all_token_ids[slot, positions.clamp_min(0)] == prefix).all()
            matches &= total_len[slot] - prompt_len[slot] >= count
            token = int(words[slot, end - 1])
            logits[row, token] = torch.where(matches, -torch.inf, logits[row, token])


def token_logprobs(out, logits, stride, ids, num_scores, vocab, **kwargs):
    values = logits.float()
    out.copy_(values.gather(1, ids.long()) - values.logsumexp(-1, keepdim=True))


def ranks(out, logits, stride, ids, vocab, **kwargs):
    out.copy_((logits >= logits.gather(1, ids.long()[:, None])).sum(-1))


def fill_logprob_ids(
    out,
    out_stride,
    mask,
    mask_stride,
    sampled,
    topk,
    topk_stride,
    mapping,
    num_custom,
    custom,
    custom_stride,
    NUM_TOPK,  # noqa: N803 - upstream Triton launch keyword
    **kwargs,
):
    slots = mapping.long()
    out[:, 0] = sampled
    mask[:, 0] = True
    cols = torch.arange(out.shape[1] - 1, device=out.device)
    counts = num_custom[slots, None]
    custom_ids = custom[slots[:, None], cols.clamp_max(custom.shape[1] - 1)]
    top_ids = (
        topk[:, cols.clamp_max(NUM_TOPK - 1)]
        if NUM_TOPK
        else torch.zeros_like(custom_ids)
    )
    valid = cols < torch.where(counts > 0, counts, NUM_TOPK)
    out[:, 1:] = torch.where(valid, torch.where(counts > 0, custom_ids, top_ids), 0)
    mask[:, 1:] = valid


def install():
    from vllm.v1.worker.gpu.sample import (
        bad_words,
        gumbel,
        logit_bias,
        logprob,
        min_p,
        penalties,
        sampler,
        states,
    )

    gumbel.gumbel_sample = sampler.gumbel_sample = gumbel_sample
    gumbel.apply_temperature = states.apply_temperature = apply_temperature
    min_p.apply_min_p = states.apply_min_p = apply_min_p
    logit_bias.apply_logit_bias = apply_logit_bias
    penalties.bincount = bincount
    penalties.apply_penalties = apply_penalties
    bad_words.apply_bad_words = apply_bad_words
    logprob._topk_log_softmax_kernel = TensorKernel(token_logprobs)
    logprob._ranks_kernel = TensorKernel(ranks)
    logprob._fill_logprob_token_ids_kernel = TensorKernel(fill_logprob_ids)
    # The retained upstream wrappers compute launch sizes on the host even
    # though their kernel entry points above now execute PyTorch operations.
    logprob.triton = SimpleNamespace(next_power_of_2=next_power_of_2)
