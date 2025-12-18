# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal configuration."""

import pytest
import torch

from vllm_metal.config import (
    MetalConfig,
    get_metal_config,
    reset_metal_config,
    set_metal_config,
)


class TestMetalConfig:
    """Tests for MetalConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetalConfig()

        assert config.device_id == 0
        assert 0 < config.memory_fraction <= 1.0
        assert config.attention_backend == "metal"
        assert config.eager_mode is True  # Always True for Metal
        assert isinstance(config.compile, bool)
        assert config.max_batch_size > 0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetalConfig(
            device_id=0,
            memory_fraction=0.8,
            attention_backend="metal",
            eager_mode=True,
            max_batch_size=128,
        )

        assert config.device_id == 0
        assert config.memory_fraction == 0.8
        assert config.attention_backend == "metal"
        assert config.eager_mode is True
        assert config.max_batch_size == 128

    def test_invalid_memory_fraction(self):
        """Test invalid memory fraction raises error."""
        with pytest.raises(ValueError, match="memory_fraction"):
            MetalConfig(memory_fraction=1.5)

        with pytest.raises(ValueError, match="memory_fraction"):
            MetalConfig(memory_fraction=0)

    def test_invalid_attention_backend(self):
        """Test invalid attention backend raises error."""
        with pytest.raises(ValueError, match="attention_backend"):
            MetalConfig(attention_backend="invalid")

        # "eager" is no longer valid - only "metal" (Rust kernels)
        with pytest.raises(ValueError, match="attention_backend"):
            MetalConfig(attention_backend="eager")

    def test_invalid_max_batch_size(self):
        """Test invalid max batch size raises error."""
        with pytest.raises(ValueError, match="max_batch_size"):
            MetalConfig(max_batch_size=0)

    def test_get_kv_cache_dtype_default(self):
        """Test default KV cache dtype."""
        config = MetalConfig()
        dtype = config.get_kv_cache_dtype()
        assert dtype == torch.float16

    def test_get_kv_cache_dtype_custom(self):
        """Test custom KV cache dtype."""
        config = MetalConfig(kv_cache_dtype="float32")
        dtype = config.get_kv_cache_dtype()
        assert dtype == torch.float32

    def test_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = MetalConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "device_id" in d
        assert "memory_fraction" in d
        assert "attention_backend" in d
        assert "eager_mode" in d
        assert "max_batch_size" in d


class TestGlobalConfig:
    """Tests for global configuration management."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_metal_config()

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_metal_config()

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_metal_config()
        assert isinstance(config, MetalConfig)

    def test_set_config(self):
        """Test setting custom configuration."""
        custom = MetalConfig(memory_fraction=0.5)
        set_metal_config(custom)

        config = get_metal_config()
        assert config.memory_fraction == 0.5

    def test_reset_config(self):
        """Test resetting configuration."""
        custom = MetalConfig(memory_fraction=0.5)
        set_metal_config(custom)

        reset_metal_config()

        config = get_metal_config()
        assert config.memory_fraction != 0.5  # Should be default
