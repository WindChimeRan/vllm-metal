# Benchmark reference from vllm-metal commit 09fbdaada935f74de7f4e9bd5d38c395041bdc9e.
# Upstream arithmetic: https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/sample/gumbel.py
# SPDX-License-Identifier: Apache-2.0
"""PyTorch implementations of MRv2 sampling kernels on MPS."""

# Kernel launch keywords retain the upstream Triton signatures.
# ruff: noqa: N803

import torch


def _right32(value, shift):
    # Logical right shift for PyTorch's signed int32 tensors.
    return (value >> shift) & ((1 << (32 - shift)) - 1)


def _rotate32(value, shift):
    return (value << shift) | _right32(value, 32 - shift)


def _mix32(h, key):
    key = _rotate32(key * 0xCC9E2D51, 15) * 0x1B873593
    return _rotate32(h ^ key, 13) * 5 + 0xE6546B64


def _uniform32(seed, position, tokens):
    # Match MRv2's Murmur3 stream, keyed by request, position and token.
    h = _mix32(torch.zeros_like(seed, dtype=torch.int32), seed.int())
    h = _mix32(h, (seed >> 32).int())
    h = _mix32(h, position.int())
    h = _mix32(h, tokens) ^ 16
    h = (h ^ _right32(h, 16)) * 0x85EBCA6B
    h = (h ^ _right32(h, 13)) * 0xC2B2AE35
    h = h ^ _right32(h, 16)
    return _right32(h, 16).float() * 2**-16 + ((h & 0xFFFF).float() + 0.5) * 2**-32


def temperature(logits, stride, mapping, values, vocab_size, BLOCK_SIZE):
    temp = values[mapping.long(), None]
    logits.div_(torch.where(temp == 0, 1, temp))


def min_p(logits, stride, mapping, values, vocab_size, BLOCK_SIZE):
    threshold = logits.amax(-1, keepdim=True) + values[mapping.long(), None].log()
    logits.masked_fill_(logits < threshold, -float("inf"))


def gumbel_sample(
    argmax,
    argmax_stride,
    maximum,
    maximum_stride,
    logits_cache,
    cache_stride,
    cache_col_stride,
    cache_col,
    cache_source,
    cache_source_stride,
    logits,
    logits_stride,
    mapping,
    seeds,
    positions,
    temperatures,
    vocab_size,
    BLOCK_SIZE,
    IS_DRAFTING,
    APPLY_TEMPERATURE,
    USE_FP64,
    PER_TOKEN_COL,
):
    if USE_FP64:
        raise NotImplementedError("MPS does not support FP64 sampling")
    rows = logits.shape[0]
    slots = mapping.long()
    source = logits if cache_source is None else cache_source
    # Cache values must retain their original dtype, including draft BF16 logits.
    if logits_cache is not None:
        cols = cache_col.expand(rows).long()
        valid = slots >= 0
        logits_cache[slots[valid], cols[valid], :vocab_size] = source[valid]
    temp = torch.where(slots >= 0, temperatures[slots], 0)[:, None]
    values = logits.float()
    if APPLY_TEMPERATURE:
        values = values / torch.where(temp == 0, 1, temp)
    tokens = torch.arange(vocab_size, device=logits.device, dtype=torch.int32)
    pos = positions[:, None] + ((1 << 30) if IS_DRAFTING else 0)
    uniform = _uniform32(seeds[slots, None], pos, tokens)
    noise = -torch.log(-torch.log1p(-uniform.clamp_max(1 - 2**-24)))
    values = torch.where(temp == 0, values, values + noise)
    padded = torch.nn.functional.pad(
        values, (0, argmax.shape[1] * BLOCK_SIZE - vocab_size), value=-float("inf")
    )
    block_max, block_idx = padded.view(rows, -1, BLOCK_SIZE).max(-1)
    maximum.copy_(block_max)
    argmax.copy_(
        block_idx + torch.arange(argmax.shape[1], device=logits.device) * BLOCK_SIZE
    )
