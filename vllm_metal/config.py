# SPDX-License-Identifier: Apache-2.0
"""Configuration for vLLM Metal backend."""

from dataclasses import dataclass, field

import torch

from vllm_metal.envs import (
    VLLM_METAL_ATTENTION_BACKEND,
    VLLM_METAL_COMPILE,
    VLLM_METAL_DEVICE_ID,
    VLLM_METAL_EAGER_MODE,
    VLLM_METAL_ENABLE_PROFILING,
    VLLM_METAL_KV_CACHE_DTYPE,
    VLLM_METAL_MAX_BATCH_SIZE,
    VLLM_METAL_MEMORY_FRACTION,
)


@dataclass
class MetalConfig:
    """Configuration for Metal backend.

    This class holds Metal-specific configuration options
    that can be set via environment variables or programmatically.
    """

    # Device configuration
    device_id: int = field(default_factory=lambda: VLLM_METAL_DEVICE_ID)

    # Memory configuration
    memory_fraction: float = field(default_factory=lambda: VLLM_METAL_MEMORY_FRACTION)

    # Backend selection
    attention_backend: str = field(default_factory=lambda: VLLM_METAL_ATTENTION_BACKEND)

    # Execution mode
    eager_mode: bool = field(default_factory=lambda: VLLM_METAL_EAGER_MODE)
    compile: bool = field(default_factory=lambda: VLLM_METAL_COMPILE)

    # Profiling
    enable_profiling: bool = field(default_factory=lambda: VLLM_METAL_ENABLE_PROFILING)

    # Limits
    max_batch_size: int = field(default_factory=lambda: VLLM_METAL_MAX_BATCH_SIZE)

    # KV cache
    kv_cache_dtype: str | None = field(
        default_factory=lambda: VLLM_METAL_KV_CACHE_DTYPE
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        if not 0 < self.memory_fraction <= 1.0:
            raise ValueError(
                f"memory_fraction must be between 0 and 1, got {self.memory_fraction}"
            )

        if self.attention_backend != "metal":
            raise ValueError(
                f"attention_backend must be 'metal' (Rust kernels), "
                f"got {self.attention_backend}"
            )

        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be positive, got {self.max_batch_size}"
            )

    def get_kv_cache_dtype(self) -> torch.dtype:
        """Get the KV cache dtype as a torch dtype."""
        if self.kv_cache_dtype is None:
            return torch.float16
        dtype: torch.dtype = getattr(torch, self.kv_cache_dtype)
        return dtype

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "device_id": self.device_id,
            "memory_fraction": self.memory_fraction,
            "attention_backend": self.attention_backend,
            "eager_mode": self.eager_mode,
            "compile": self.compile,
            "enable_profiling": self.enable_profiling,
            "max_batch_size": self.max_batch_size,
            "kv_cache_dtype": self.kv_cache_dtype,
        }


# Global configuration instance
_metal_config: MetalConfig | None = None


def get_metal_config() -> MetalConfig:
    """Get the global Metal configuration.

    Returns:
        MetalConfig instance
    """
    global _metal_config
    if _metal_config is None:
        _metal_config = MetalConfig()
    return _metal_config


def set_metal_config(config: MetalConfig) -> None:
    """Set the global Metal configuration.

    Args:
        config: MetalConfig instance to use
    """
    global _metal_config
    _metal_config = config


def reset_metal_config() -> None:
    """Reset Metal configuration to defaults."""
    global _metal_config
    _metal_config = None
