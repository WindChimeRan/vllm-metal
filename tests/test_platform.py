# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal platform."""

import pytest
import torch

from vllm_metal._compat import PlatformEnum
from vllm_metal.platform import MetalPlatform


class TestMetalPlatform:
    """Tests for MetalPlatform class."""

    def test_platform_enum(self):
        """Test platform enum is OOT (out-of-tree)."""
        assert MetalPlatform._enum == PlatformEnum.OOT

    def test_device_name(self):
        """Test device name is 'mps' (PyTorch Metal device)."""
        assert MetalPlatform.device_name == "mps"
        assert MetalPlatform.device_type == "mps"

    def test_dispatch_key(self):
        """Test dispatch key is 'MPS' (PyTorch dispatch key)."""
        assert MetalPlatform.dispatch_key == "MPS"

    def test_supported_quantization(self):
        """Test supported quantization methods."""
        assert "awq" in MetalPlatform.supported_quantization
        assert "gptq" in MetalPlatform.supported_quantization

    @pytest.mark.metal
    def test_get_device_name(self):
        """Test getting device name."""
        name = MetalPlatform.get_device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    @pytest.mark.metal
    def test_get_device_uuid(self):
        """Test getting device UUID."""
        uuid = MetalPlatform.get_device_uuid()
        assert isinstance(uuid, str)
        assert uuid.startswith("metal:")

    @pytest.mark.metal
    def test_get_device_total_memory(self):
        """Test getting device memory."""
        memory = MetalPlatform.get_device_total_memory()
        assert isinstance(memory, int)
        assert memory > 0

    @pytest.mark.metal
    def test_inference_mode(self):
        """Test inference mode context manager."""
        ctx = MetalPlatform.inference_mode()
        assert ctx is not None

        with ctx:
            # PyTorch uses "mps" device type for Metal
            x = torch.randn(10, 10, device="mps")
            y = x + 1
            assert y.shape == x.shape

    @pytest.mark.metal
    def test_seed_everything(self):
        """Test seeding random generators."""
        MetalPlatform.seed_everything(42)

        # Generate tensors with same seed
        # PyTorch uses "mps" device type for Metal
        MetalPlatform.seed_everything(42)
        a = torch.randn(10, device="mps")

        MetalPlatform.seed_everything(42)
        b = torch.randn(10, device="mps")

        torch.testing.assert_close(a, b)

    @pytest.mark.metal
    def test_empty_cache(self):
        """Test emptying cache."""
        # Create some tensors (PyTorch uses "mps" device type for Metal)
        _ = torch.randn(100, 100, device="mps")

        # Should not raise
        MetalPlatform.empty_cache()

    @pytest.mark.metal
    def test_synchronize(self):
        """Test synchronization."""
        # PyTorch uses "mps" device type for Metal
        x = torch.randn(100, 100, device="mps")
        _ = x @ x.T

        # Should not raise
        MetalPlatform.synchronize()

    @pytest.mark.metal
    def test_mem_get_info(self):
        """Test getting memory info."""
        free, total = MetalPlatform.mem_get_info()

        assert isinstance(free, int)
        assert isinstance(total, int)
        assert total > 0

    def test_is_pin_memory_available(self):
        """Test pin memory availability."""
        # Metal doesn't support pinned memory
        assert MetalPlatform.is_pin_memory_available() is False

    def test_supports_fp8(self):
        """Test FP8 support."""
        # Metal doesn't support FP8
        assert MetalPlatform.supports_fp8() is False

    def test_supports_mx(self):
        """Test MX format support."""
        # Metal doesn't support MX formats
        assert MetalPlatform.supports_mx() is False

    def test_check_if_supports_dtype(self):
        """Test dtype support checking."""
        assert MetalPlatform.check_if_supports_dtype(torch.float16) is True
        assert MetalPlatform.check_if_supports_dtype(torch.float32) is True
        assert MetalPlatform.check_if_supports_dtype(torch.bfloat16) is True
        assert MetalPlatform.check_if_supports_dtype(torch.int32) is True

    def test_verify_quantization_valid(self):
        """Test valid quantization verification."""
        # Should not raise
        MetalPlatform.verify_quantization("awq")
        MetalPlatform.verify_quantization("gptq")
        MetalPlatform.verify_quantization(None)

    def test_verify_quantization_invalid(self):
        """Test invalid quantization verification."""
        with pytest.raises(ValueError, match="not supported"):
            MetalPlatform.verify_quantization("fp8")

    def test_support_hybrid_kv_cache(self):
        """Test hybrid KV cache support."""
        assert MetalPlatform.support_hybrid_kv_cache() is False

    def test_support_static_graph_mode(self):
        """Test static graph mode support."""
        assert MetalPlatform.support_static_graph_mode() is False

    def test_can_update_inplace(self):
        """Test in-place update support."""
        assert MetalPlatform.can_update_inplace() is True

    def test_get_attn_backend_cls(self):
        """Test getting attention backend class."""
        import torch

        cls = MetalPlatform.get_attn_backend_cls(
            selected_backend=None,
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            attn_type=None,
        )
        assert cls == "vllm_metal.attention.backend.MetalAttentionBackend"

    def test_get_device_communicator_cls(self):
        """Test getting communicator class."""
        # Metal doesn't support distributed
        assert MetalPlatform.get_device_communicator_cls() is None

    def test_stateless_init_distributed_raises(self):
        """Test that distributed init raises."""
        with pytest.raises(NotImplementedError):
            MetalPlatform.stateless_init_device_torch_dist_pg("gloo", 60)
