# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal attention backend."""

import pytest
import torch
from torch.nn import functional

from vllm_metal.attention.backend import (
    MetalAttentionBackend,
    MetalAttentionImpl,
    MetalAttentionMetadata,
)


class TestMetalAttentionBackend:
    """Tests for MetalAttentionBackend."""

    def test_get_name(self):
        """Test backend name."""
        assert MetalAttentionBackend.get_name() == "METAL"

    def test_get_impl_cls(self):
        """Test getting implementation class."""
        impl_cls = MetalAttentionBackend.get_impl_cls()
        assert impl_cls is MetalAttentionImpl

    def test_get_metadata_cls(self):
        """Test getting metadata class."""
        meta_cls = MetalAttentionBackend.get_metadata_cls()
        assert meta_cls is MetalAttentionMetadata

    def test_get_kv_cache_shape(self):
        """Test KV cache shape calculation."""
        num_blocks = 100
        block_size = 16
        num_kv_heads = 8
        head_size = 64

        shape = MetalAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )

        assert shape == (num_blocks, 2, block_size, num_kv_heads, head_size)

    def test_get_supported_head_sizes(self):
        """Test supported head sizes."""
        sizes = MetalAttentionBackend.get_supported_head_sizes()

        assert isinstance(sizes, list)
        assert 64 in sizes
        assert 128 in sizes


class TestMetalAttentionMetadata:
    """Tests for MetalAttentionMetadata (V1 API)."""

    def test_metadata_creation(self, metal_device):
        """Test creating metadata with V1 parameters."""
        seq_len = 16
        meta = MetalAttentionMetadata(
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            query_start_loc=torch.tensor([0, seq_len], device=metal_device),
            max_seq_len=seq_len,
            seq_lens=torch.tensor([seq_len], device=metal_device),
            block_table=torch.zeros((1, 1), dtype=torch.int32, device=metal_device),
            slot_mapping=torch.arange(seq_len, device=metal_device),
        )

        assert meta.num_actual_tokens == seq_len
        assert meta.max_query_len == seq_len
        assert meta.max_seq_len == seq_len
        assert meta.use_cascade is False

    def test_prefill_metadata(self, metal_device):
        """Test metadata for prefill (query_len > 1)."""
        seq_len = 32
        meta = MetalAttentionMetadata(
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            query_start_loc=torch.tensor([0, seq_len], device=metal_device),
            max_seq_len=seq_len,
            seq_lens=torch.tensor([seq_len], device=metal_device),
            block_table=torch.zeros((1, 2), dtype=torch.int32, device=metal_device),
            slot_mapping=torch.arange(seq_len, device=metal_device),
        )

        # Prefill is when max_query_len > 1
        assert meta.max_query_len > 1

    def test_decode_metadata(self, metal_device):
        """Test metadata for decode (query_len == 1)."""
        batch_size = 4
        meta = MetalAttentionMetadata(
            num_actual_tokens=batch_size,
            max_query_len=1,  # Decode has query_len of 1
            query_start_loc=torch.arange(batch_size + 1, device=metal_device),
            max_seq_len=100,
            seq_lens=torch.full((batch_size,), 100, device=metal_device),
            block_table=torch.zeros(
                (batch_size, 7), dtype=torch.int32, device=metal_device
            ),
            slot_mapping=torch.arange(batch_size, device=metal_device),
        )

        # Decode is when max_query_len == 1
        assert meta.max_query_len == 1


class TestMetalAttentionImpl:
    """Tests for Metal attention implementation."""

    @pytest.fixture
    def attention_impl(self):
        """Create attention implementation for testing."""
        return MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0 / (64**0.5),
            num_kv_heads=8,
        )

    @pytest.mark.metal
    def test_init(self, attention_impl):
        """Test attention initialization."""
        assert attention_impl.num_heads == 8
        assert attention_impl.head_size == 64
        assert attention_impl.num_kv_heads == 8

    @pytest.mark.metal
    def test_forward_prefill(self, attention_impl, metal_device):
        """Test forward pass for prefill."""
        seq_len = 16
        num_heads = 8
        head_size = 64

        query = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float16
        )
        key = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float16
        )
        value = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float16
        )

        metadata = MetalAttentionMetadata(
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            query_start_loc=torch.tensor([0, seq_len], device=metal_device),
            max_seq_len=seq_len,
            seq_lens=torch.tensor([seq_len], device=metal_device),
            block_table=torch.zeros((1, 1), dtype=torch.int32, device=metal_device),
            slot_mapping=torch.arange(seq_len, device=metal_device),
        )

        output = attention_impl.forward(
            layer=None,  # Not used in Metal impl
            query=query,
            key=key,
            value=value,
            kv_cache=None,
            attn_metadata=metadata,
        )

        assert output.shape == (seq_len, num_heads * head_size)

    @pytest.mark.metal
    def test_gqa_support(self, metal_device):
        """Test grouped query attention support."""
        num_heads = 32
        num_kv_heads = 8  # GQA with 4 query groups
        head_size = 64

        impl = MetalAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 / (head_size**0.5),
            num_kv_heads=num_kv_heads,
        )

        seq_len = 16
        query = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float16
        )
        key = torch.randn(
            seq_len, num_kv_heads * head_size, device=metal_device, dtype=torch.float16
        )
        value = torch.randn(
            seq_len, num_kv_heads * head_size, device=metal_device, dtype=torch.float16
        )

        metadata = MetalAttentionMetadata(
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            query_start_loc=torch.tensor([0, seq_len], device=metal_device),
            max_seq_len=seq_len,
            seq_lens=torch.tensor([seq_len], device=metal_device),
            block_table=torch.zeros((1, 1), dtype=torch.int32, device=metal_device),
            slot_mapping=torch.arange(seq_len, device=metal_device),
        )

        output = impl.forward(
            layer=None,  # Not used in Metal impl
            query=query,
            key=key,
            value=value,
            kv_cache=None,
            attn_metadata=metadata,
        )

        assert output.shape == (seq_len, num_heads * head_size)


class TestAttentionCorrectness:
    """Tests for attention correctness."""

    @pytest.mark.metal
    def test_attention_matches_reference(self, metal_device):
        """Test that attention output matches reference implementation."""
        num_heads = 4
        head_size = 32
        seq_len = 8

        impl = MetalAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 / (head_size**0.5),
        )

        # Create inputs
        query = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float32
        )
        key = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float32
        )
        value = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float32
        )

        metadata = MetalAttentionMetadata(
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            query_start_loc=torch.tensor([0, seq_len], device=metal_device),
            max_seq_len=seq_len,
            seq_lens=torch.tensor([seq_len], device=metal_device),
            block_table=torch.zeros((1, 1), dtype=torch.int32, device=metal_device),
            slot_mapping=torch.arange(seq_len, device=metal_device),
        )

        # Our implementation
        output = impl.forward(
            layer=None,  # Not used in Metal impl
            query=query,
            key=key,
            value=value,
            kv_cache=None,
            attn_metadata=metadata,
        )

        # Reference: manual scaled dot product attention
        q = query.view(seq_len, num_heads, head_size).transpose(0, 1)
        k = key.view(seq_len, num_heads, head_size).transpose(0, 1)
        v = value.view(seq_len, num_heads, head_size).transpose(0, 1)

        scale = 1.0 / (head_size**0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=metal_device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = functional.softmax(attn_weights, dim=-1)
        expected = torch.matmul(attn_weights, v)
        expected = expected.transpose(0, 1).reshape(seq_len, num_heads * head_size)

        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)
