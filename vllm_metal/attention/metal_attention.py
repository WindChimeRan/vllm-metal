# SPDX-License-Identifier: Apache-2.0
"""Metal attention implementation for vLLM Metal backend."""

from typing import Any

import torch
from torch.nn import functional

from vllm_metal._compat import AttentionImpl, AttentionType, init_logger
from vllm_metal.attention.backend import MetalAttentionMetadata

logger = init_logger(__name__)


class MetalAttentionImpl(AttentionImpl):
    """Metal-based attention implementation.

    This implementation uses PyTorch's scaled_dot_product_attention
    which is optimized for Apple Silicon via MPS backend.

    Note: Custom Metal kernels (rust_ext/shaders/sdpa_vector.metal) are
    compiled but not yet integrated. The SDPA kernel dispatch in
    rust_ext/src/metal/kernels/attention.rs is a stub.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: dict[str, Any] | None = None,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        **kwargs,
    ) -> None:
        """Initialize Metal attention.

        Args:
            num_heads: Number of query attention heads.
            head_size: Size of each attention head.
            scale: Scaling factor for attention scores.
            num_kv_heads: Number of key/value attention heads.
            alibi_slopes: ALiBi slopes for position encoding.
            sliding_window: Sliding window size for attention.
            kv_cache_dtype: Data type for KV cache.
            blocksparse_params: Block sparse attention parameters.
            logits_soft_cap: Soft cap for attention logits.
            attn_type: Type of attention (decoder, encoder, etc.).
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.alibi_slopes = alibi_slopes
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type

        # Calculate number of query groups for GQA
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.alibi_slopes_tensor: torch.Tensor | None
        if alibi_slopes is not None:
            self.alibi_slopes_tensor = torch.tensor(alibi_slopes, dtype=torch.float32)
        else:
            self.alibi_slopes_tensor = None

        if blocksparse_params is not None:
            logger.warning(
                "Block sparse attention is not supported on Metal, "
                "falling back to dense attention"
            )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Process weights after model loading.

        This is called after the model weights are loaded to perform
        any necessary post-processing. For Metal attention, we just
        ensure alibi slopes are on the correct device/dtype.
        """
        if self.alibi_slopes_tensor is not None:
            self.alibi_slopes_tensor = self.alibi_slopes_tensor.to(dtype=act_dtype)

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | None,
        attn_metadata: MetalAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        output: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for attention.

        Args:
            layer: The Attention layer calling this impl (vLLM interface requirement)
            query: Query tensor of shape [num_tokens, num_heads * head_size]
            key: Key tensor of shape [num_tokens, num_kv_heads * head_size]
            value: Value tensor of shape [num_tokens, num_kv_heads * head_size]
            kv_cache: KV cache tensor
            attn_metadata: Attention metadata
            k_scale: Key scaling factor
            v_scale: Value scaling factor
            output: Optional output tensor to write to

        Returns:
            Output tensor of shape [num_tokens, num_heads * head_size]
        """
        num_tokens = query.shape[0]

        # Reshape query, key, value for attention
        # [num_tokens, num_heads, head_size]
        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        # Handle prefill and decode separately
        if attn_metadata.is_prompt and attn_metadata.num_prefill_tokens > 0:
            # Prefill phase
            out = self._prefill_attention(query, key, value, kv_cache, attn_metadata)
        else:
            # Decode phase
            out = self._decode_attention(query, key, value, kv_cache, attn_metadata)

        # Reshape output to [num_tokens, num_heads * head_size]
        out = out.view(num_tokens, self.num_heads * self.head_size)

        if output is not None:
            output.copy_(out)
            return output

        return out

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | None,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Attention computation for prefill phase.

        Uses PyTorch's scaled_dot_product_attention for efficiency.
        """
        # Store KV to cache if available
        if kv_cache is not None and attn_metadata.slot_mapping is not None:
            self._store_kv_cache(key, value, kv_cache, attn_metadata.slot_mapping)

        # Process each sequence in the batch
        if attn_metadata.seq_lens is None:
            # Single sequence case
            return self._compute_attention(query, key, value, is_causal=True)

        # Multiple sequences with variable lengths
        outputs = []
        start_idx = 0

        for seq_len in attn_metadata.seq_lens:
            q = query[start_idx : start_idx + seq_len]
            k = key[start_idx : start_idx + seq_len]
            v = value[start_idx : start_idx + seq_len]

            out = self._compute_attention(q, k, v, is_causal=True)
            outputs.append(out)
            start_idx += seq_len

        return torch.cat(outputs, dim=0)

    def _decode_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | None,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Attention computation for decode phase.

        Uses paged attention with KV cache.
        """
        if kv_cache is None:
            raise ValueError("KV cache is required for decode attention")

        # Store new KV to cache
        if attn_metadata.slot_mapping is not None:
            self._store_kv_cache(key, value, kv_cache, attn_metadata.slot_mapping)

        # Read from paged KV cache
        batch_size = query.shape[0]
        outputs = []

        for i in range(batch_size):
            # Get block table for this sequence
            if attn_metadata.block_tables is not None:
                block_table = attn_metadata.block_tables[i]
            else:
                continue

            assert attn_metadata.context_lens_tensor is not None
            context_len = int(attn_metadata.context_lens_tensor[i].item()) + 1

            # Gather keys and values from cache
            k_cache, v_cache = self._gather_from_cache(
                kv_cache, block_table, context_len
            )

            # Single query token attention against cached KV
            q = query[i : i + 1]  # [1, num_heads, head_size]

            # Expand KV for GQA if needed
            if self.num_queries_per_kv > 1:
                k_cache = k_cache.repeat_interleave(self.num_queries_per_kv, dim=1)
                v_cache = v_cache.repeat_interleave(self.num_queries_per_kv, dim=1)

            out = self._compute_attention(q, k_cache, v_cache, is_causal=False)
            outputs.append(out)

        return torch.cat(outputs, dim=0)

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Uses PyTorch's optimized SDPA which works well on Metal.
        """
        # Input shapes: [seq_len, num_heads, head_size]
        # SDPA expects: [batch, num_heads, seq_len, head_size]
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)

        # Expand KV for GQA if needed
        if self.num_queries_per_kv > 1 and key.shape[1] != query.shape[1]:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Apply attention with optional sliding window
        if self.sliding_window is not None and is_causal:
            # Create sliding window mask
            seq_len = query.shape[2]
            attn_mask = self._create_sliding_window_mask(
                seq_len, self.sliding_window, query.device, query.dtype
            )
            is_causal = False

        # Apply alibi slopes if configured
        if self.alibi_slopes_tensor is not None:
            attn_mask = self._apply_alibi(
                query.shape[2],
                key.shape[2],
                query.device,
                query.dtype,
                attn_mask,
            )
            is_causal = False

        # Use PyTorch's SDPA
        out = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        )

        # Apply logits soft cap if needed
        if self.logits_soft_cap is not None:
            # Note: SDPA doesn't support soft cap directly
            # This would need custom implementation
            pass

        # Reshape back: [batch, num_heads, seq_len, head_size] -> [seq_len, num_heads, head_size]
        out = out.squeeze(0).transpose(0, 1)

        return out

    def _store_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store key and value tensors into the KV cache.

        KV cache shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
        """
        num_tokens = key.shape[0]

        for i in range(num_tokens):
            slot = int(slot_mapping[i].item())
            block_idx = slot // kv_cache.shape[2]
            block_offset = slot % kv_cache.shape[2]

            # Store key
            kv_cache[block_idx, 0, block_offset] = key[i]
            # Store value
            kv_cache[block_idx, 1, block_offset] = value[i]

    def _gather_from_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather keys and values from the paged KV cache.

        Args:
            kv_cache: KV cache tensor
            block_table: Block indices for this sequence
            context_len: Context length to gather

        Returns:
            Tuple of (keys, values) tensors
        """
        block_size = kv_cache.shape[2]

        # Calculate number of blocks needed
        num_blocks_needed = (context_len + block_size - 1) // block_size

        keys = []
        values = []

        tokens_gathered = 0
        for block_idx in range(num_blocks_needed):
            if block_idx >= len(block_table):
                break

            physical_block = int(block_table[block_idx].item())
            tokens_in_block = min(block_size, context_len - tokens_gathered)

            # Gather from this block
            k_block = kv_cache[physical_block, 0, :tokens_in_block]
            v_block = kv_cache[physical_block, 1, :tokens_in_block]

            keys.append(k_block)
            values.append(v_block)
            tokens_gathered += tokens_in_block

        # Concatenate all blocks
        keys_tensor = torch.cat(keys, dim=0)  # [context_len, num_kv_heads, head_size]
        values_tensor = torch.cat(values, dim=0)

        return keys_tensor, values_tensor

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a sliding window attention mask."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            mask[i, start : i + 1] = 0
        return mask

    def _apply_alibi(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
        dtype: torch.dtype,
        existing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply ALiBi position bias to attention mask."""
        assert self.alibi_slopes_tensor is not None, "ALiBi slopes must be set"
        alibi_slopes = self.alibi_slopes_tensor.to(device=device, dtype=dtype)

        # Create position bias
        # Shape: [num_heads, query_len, key_len]
        positions = torch.arange(key_len, device=device, dtype=dtype)
        query_positions = torch.arange(query_len, device=device, dtype=dtype)

        # Distance matrix
        distances = query_positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = distances.clamp(max=0)  # Only attend to past positions

        # Apply slopes
        alibi_bias = alibi_slopes.unsqueeze(1).unsqueeze(2) * distances.unsqueeze(0)

        if existing_mask is not None:
            alibi_bias = alibi_bias + existing_mask

        return alibi_bias
