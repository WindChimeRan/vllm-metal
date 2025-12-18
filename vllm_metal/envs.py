# SPDX-License-Identifier: Apache-2.0
"""Environment variable configuration for vLLM Metal backend."""

import os

# Metal-specific environment variables
VLLM_METAL_DEVICE_ID: int = int(os.getenv("VLLM_METAL_DEVICE_ID", "0"))

# Memory configuration
VLLM_METAL_MEMORY_FRACTION: float = float(
    os.getenv("VLLM_METAL_MEMORY_FRACTION", "0.9")
)

# Attention backend selection (always "metal" for Rust kernels)
VLLM_METAL_ATTENTION_BACKEND: str = os.getenv("VLLM_METAL_ATTENTION_BACKEND", "metal")

# Enable profiling
VLLM_METAL_ENABLE_PROFILING: bool = os.getenv(
    "VLLM_METAL_ENABLE_PROFILING", "0"
).lower() in ("1", "true", "yes")

# Compilation settings
VLLM_METAL_COMPILE: bool = os.getenv("VLLM_METAL_COMPILE", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Batch size limits for Metal
VLLM_METAL_MAX_BATCH_SIZE: int = int(os.getenv("VLLM_METAL_MAX_BATCH_SIZE", "256"))

# KV cache dtype override (None means use model dtype)
VLLM_METAL_KV_CACHE_DTYPE: str | None = os.getenv("VLLM_METAL_KV_CACHE_DTYPE", None)

# Metal uses custom kernels (no eager PyTorch fallback)
# This is kept for vLLM compatibility but always True
VLLM_METAL_EAGER_MODE: bool = True


def get_metal_env_info() -> dict:
    """Get all Metal environment configuration as a dictionary."""
    return {
        "device_id": VLLM_METAL_DEVICE_ID,
        "memory_fraction": VLLM_METAL_MEMORY_FRACTION,
        "attention_backend": VLLM_METAL_ATTENTION_BACKEND,
        "enable_profiling": VLLM_METAL_ENABLE_PROFILING,
        "compile": VLLM_METAL_COMPILE,
        "max_batch_size": VLLM_METAL_MAX_BATCH_SIZE,
        "kv_cache_dtype": VLLM_METAL_KV_CACHE_DTYPE,
        "eager_mode": VLLM_METAL_EAGER_MODE,
    }
