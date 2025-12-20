# SPDX-License-Identifier: Apache-2.0
"""Batched KV Cache utilities for MLX-based inference.

Provides utilities for efficient batch processing of variable-length sequences.
"""

import mlx.core as mx


def create_left_padded_batch(
    token_sequences: list[list[int]], pad_token: int = 0
) -> tuple[mx.array, list[int]]:
    """Create a left-padded batch from variable-length sequences.

    Left-padding is required by BatchKVCache to properly align sequences
    of different lengths for batched attention.

    Args:
        token_sequences: List of token ID sequences
        pad_token: Token to use for padding (default: 0)

    Returns:
        Tuple of (padded_batch, left_padding_amounts)
        - padded_batch: Shape (batch_size, max_length)
        - left_padding_amounts: How much each sequence was padded
    """
    if not token_sequences:
        return mx.array([], dtype=mx.int32), []

    max_length = max(len(seq) for seq in token_sequences)

    padded = []
    padding_amounts = []

    for seq in token_sequences:
        pad_amount = max_length - len(seq)
        padding_amounts.append(pad_amount)
        padded_seq = [pad_token] * pad_amount + list(seq)
        padded.append(padded_seq)

    return mx.array(padded, dtype=mx.int32), padding_amounts


def group_by_length(
    sequences: list[tuple[str, list[int]]],
    max_padding_ratio: float = 0.2,
    max_batch_size: int = 32,
) -> list[list[tuple[str, list[int]]]]:
    """Group sequences by similar length for efficient batching.

    Minimizes padding waste by grouping sequences of similar length together.
    This is particularly useful for batched prefill where prompts vary in length.

    Args:
        sequences: List of (req_id, token_ids) tuples
        max_padding_ratio: Maximum ratio of padding to sequence length (default: 0.2)
        max_batch_size: Maximum sequences per batch (default: 32)

    Returns:
        List of batches, where each batch contains sequences of similar length
    """
    if not sequences:
        return []

    # Sort by length for optimal grouping
    sorted_seqs = sorted(sequences, key=lambda x: len(x[1]))

    batches: list[list[tuple[str, list[int]]]] = []
    current_batch: list[tuple[str, list[int]]] = []
    current_max_len = 0

    for req_id, tokens in sorted_seqs:
        seq_len = len(tokens)

        if current_batch:
            # Check if adding this sequence would exceed padding ratio
            new_max_len = max(current_max_len, seq_len)
            min_len_in_batch = len(current_batch[0][1])
            padding_ratio = (new_max_len - min_len_in_batch) / max(min_len_in_batch, 1)

            if (
                padding_ratio > max_padding_ratio
                or len(current_batch) >= max_batch_size
            ):
                batches.append(current_batch)
                current_batch = []
                current_max_len = 0

        current_batch.append((req_id, tokens))
        current_max_len = max(current_max_len, seq_len)

    if current_batch:
        batches.append(current_batch)

    return batches
