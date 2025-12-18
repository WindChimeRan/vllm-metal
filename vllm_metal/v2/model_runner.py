# SPDX-License-Identifier: Apache-2.0
"""Metal V2 Model Runner - extends GPU model runner for Metal/MPS backend."""

from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# ============================================================================
# Module-level patching for Metal (must happen before importing GPUModelRunner)
# ============================================================================


def _patched_bincount_metal(
    prefill_token_ids: torch.Tensor,
    prefill_len: int,
    prompt_len: int,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """PyTorch-based bincount replacement for Metal (no Triton)."""
    prompt_bin_mask.zero_()
    output_bin_counts.zero_()

    # Get the tokens in the range [prompt_len, prefill_len)
    if prefill_len > prompt_len:
        tokens = prefill_token_ids[prompt_len:prefill_len]
        # Use PyTorch's bincount
        # Move to CPU for bincount if needed, then copy back
        tokens_cpu = tokens.cpu().to(torch.int64)
        vocab_size = output_bin_counts.shape[0]
        counts = torch.bincount(tokens_cpu, minlength=vocab_size)
        # Copy counts to output (truncate if vocab sizes differ)
        min_len = min(len(counts), vocab_size)
        output_bin_counts[:min_len] = counts[:min_len].to(output_bin_counts.device)

    # Set prompt_bin_mask for tokens in [0, prompt_len)
    if prompt_len > 0:
        prompt_tokens = prefill_token_ids[:prompt_len]
        prompt_tokens_cpu = prompt_tokens.cpu().to(torch.int64)
        vocab_size = prompt_bin_mask.shape[0]
        # Create a mask indicating which tokens appeared in the prompt
        for token in prompt_tokens_cpu:
            if 0 <= token < vocab_size:
                prompt_bin_mask[token] = 1


# Patch bincount BEFORE any vLLM modules that use it are imported
# This is critical because states.py imports bincount at module load time
try:
    import vllm.v1.worker.gpu.sample.penalties as penalties_module

    penalties_module.bincount = _patched_bincount_metal
    logger.debug("Patched penalties_module.bincount for Metal")
except ImportError:
    pass

# Patch input_batch functions BEFORE importing GPUModelRunner
# These functions use Triton kernels which are not available on Metal
try:
    import vllm.v1.worker.gpu.input_batch as input_batch_module

    from vllm_metal.v2.input_batch import (
        combine_sampled_and_draft_tokens,
        post_update,
        prepare_pos_seq_lens,
        prepare_prefill_inputs,
    )

    input_batch_module.prepare_prefill_inputs = prepare_prefill_inputs
    input_batch_module.prepare_pos_seq_lens = prepare_pos_seq_lens
    input_batch_module.combine_sampled_and_draft_tokens = (
        combine_sampled_and_draft_tokens
    )
    input_batch_module.post_update = post_update
    logger.debug("Patched input_batch module functions for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch input_batch module: {e}")

# Patch penalties module - uses Triton kernels for temperature and penalties
try:
    import vllm.v1.worker.gpu.sample.penalties as penalties_module

    from vllm_metal.v2.penalties import apply_penalties_and_temperature

    penalties_module.apply_penalties_and_temperature = apply_penalties_and_temperature
    logger.debug("Patched penalties_module.apply_penalties_and_temperature for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch penalties module: {e}")

# Patch gumbel module - uses Triton kernel for gumbel sampling
try:
    import vllm.v1.worker.gpu.sample.gumbel as gumbel_module

    from vllm_metal.v2.gumbel import gumbel_sample

    gumbel_module.gumbel_sample = gumbel_sample
    logger.debug("Patched gumbel_module.gumbel_sample for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch gumbel module: {e}")

# Patch async_utils module - uses CUDA streams for async copies
try:
    import vllm.v1.worker.gpu.async_utils as async_utils_module

    from vllm_metal.v2.async_utils import MetalAsyncOutput

    async_utils_module.AsyncOutput = MetalAsyncOutput
    logger.debug("Patched async_utils_module.AsyncOutput for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch async_utils module: {e}")

# Now import the rest of vLLM modules (they will get our patched functions)
# These must be imported after the patches above, hence E402
from vllm.model_executor.model_loader import get_model  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: E402
from vllm.v1.utils import CpuGpuBuffer  # noqa: E402
from vllm.v1.worker.gpu.attn_utils import (  # noqa: E402
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner  # noqa: E402

# Use our Metal-compatible BlockTables instead of the Triton-based one
from vllm_metal.v2.block_table import MetalBlockTables as BlockTables  # noqa: E402

# Also patch states module's bincount reference (imported before we could patch)
try:
    import vllm.v1.worker.gpu.states as states_module

    states_module.bincount = _patched_bincount_metal
    logger.debug("Patched states_module.bincount for Metal")
except (ImportError, AttributeError):
    pass

# Check for Rust Metal extensions availability
try:
    import vllm_metal_rust  # noqa: F401

    RUST_AVAILABLE = True
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"Rust Metal V2 extensions not available: {e}")


class MetalModelRunner(GPUModelRunner):
    """Metal/MPS model runner that extends the GPU model runner.

    This class inherits all the complex input batch management, attention
    metadata building, and model execution from GPUModelRunner. It only
    overrides Metal-specific functionality like:
    - Disabling CUDA-specific features (pinned memory, CUDA graphs)
    - Using MPS synchronization instead of CUDA
    - Metal-specific device handling
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Use the CUDA wrapper to prevent CUDA stream/event creation during init
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # Override CUDA-specific settings
        self.pin_memory = False  # Metal uses unified memory
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace GPU tensors with MPS equivalents
        self._postprocess_tensors()

        # Log initialization
        logger.info(
            f"MetalModelRunner V2 initialized: "
            f"hidden_size={self.model_config.get_hidden_size()}, "
            f"num_heads={self.model_config.get_num_attention_heads(self.parallel_config)}, "
            f"num_kv_heads={self.model_config.get_num_kv_heads(self.parallel_config)}, "
            f"head_dim={self.model_config.get_head_size()}, "
            f"block_size={self.cache_config.block_size}, "
            f"rust_available={RUST_AVAILABLE}"
        )

    def _postprocess_tensors(self) -> None:
        """Replace GPU tensors with device tensors for Metal."""
        # For Metal, we don't need separate CPU and GPU buffers
        # since MPS uses unified memory
        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                # For unified memory, gpu buffer can point to cpu buffer
                v.gpu = v.cpu

    def _sync_device(self) -> None:
        """Synchronize the MPS device instead of CUDA."""
        torch.mps.synchronize()

    def load_model(self, *args, **kwargs) -> None:
        """Load the model to the MPS device."""
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(
                self.model,
                self.vllm_config,
                self.device,
            )

        # Ensure model is on MPS
        if self.model is not None:
            memory_gb = torch.mps.current_allocated_memory() / 1e9
            logger.info(
                f"Model loaded on device: {self.device}, memory: {memory_gb:.2f}GB"
            )

    def get_model(self) -> nn.Module:
        return self.model

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize KV cache for Metal backend.

        Override to skip the FLASH_ATTN check that would fail for Metal backend.
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]

        logger.info(
            f"Creating BlockTables using class: {BlockTables.__module__}.{BlockTables.__name__}"
        )
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        logger.info(
            f"BlockTables created: {type(self.block_tables).__module__}.{type(self.block_tables).__name__}"
        )

        self.attn_backends, self.attn_metadata_builders = init_attn_backend(
            self.kv_cache_config,
            self.vllm_config,
            self.device,
        )
        if self.do_spec_decode:
            # HACK(woosuk)
            self.speculator.set_attn(
                self.kv_cache_config,
                self.attn_metadata_builders,
                self.block_tables,
            )

        # Skip the FLASH_ATTN check - Metal uses its own backend
        # Metal backend is initialized via platform.get_attn_backend_cls()
        for name, backend in self.attn_backends.items():
            logger.info(f"Attention backend for '{name}': {backend.get_name()}")

        self.kv_caches: list[torch.Tensor] = []
        init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_backends,
            self.device,
        )
        # Attention groups are not supported.
        self.attn_groups = []  # type: ignore

    def _maybe_get_cuda_graph_runner(self, *args, **kwargs):
        """Metal does not support CUDA graphs - always return None."""
        return None

    def capture_model(self) -> int:
        """Metal does not support CUDA graph capture."""
        logger.debug("Metal does not support graph capture, skipping")
        return 0

    def compile_or_warm_up_model(self) -> None:
        """Warm up the model without CUDA graph compilation."""
        logger.info("Warming up Metal model...")
        torch.mps.synchronize()
        logger.info("Metal model warmup complete")

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Run profiling - simplified for Metal."""
        logger.info("Running Metal V2 profiling...")
        torch.mps.synchronize()
        logger.info("Metal V2 profiling complete")

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        """Metal doesn't need DP padding."""
        return 0, None


@contextmanager
def _torch_cuda_wrapper():
    """Context manager to mock CUDA stream/event during GPUModelRunner init.

    GPUModelRunner creates torch.cuda.Stream and torch.cuda.Event objects
    which fail on non-CUDA devices. This wrapper temporarily replaces them
    with placeholder objects that do nothing.

    Also patches:
    - is_uva_available() to return True since Metal's unified memory is
      semantically similar to CUDA UVA
    - torch.zeros/torch.empty to strip pin_memory=True (not supported on MPS)
    - UvaBuffer to use Metal unified memory instead of CUDA UVA
    - get_cuda_view_from_cpu_tensor to return MPS tensor instead
    """

    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda *a, **kw: None
            self.synchronize = lambda *a, **kw: None
            self.wait = lambda *a, **kw: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def wait_event(self, *args, **kwargs) -> None:
            pass

        def wait_stream(self, *args, **kwargs) -> None:
            pass

    class _MetalUvaBuffer:
        """Metal-compatible UvaBuffer using unified memory.

        On Metal, CPU and GPU share the same memory, so we just create
        a CPU tensor and use it directly (no need for pinned memory or
        CUDA views).
        """

        def __init__(self, *size, dtype: torch.dtype):
            # Create regular CPU tensor - Metal has unified memory
            self.cpu = torch.zeros(*size, dtype=dtype, device="cpu")
            self.np = self.cpu.numpy()
            # For Metal, gpu points to an MPS view of the same data
            # Since Metal has unified memory, the MPS tensor can access
            # the same underlying memory
            self.gpu = self.cpu.to("mps")

    # Save original classes and functions
    cuda_event = torch.cuda.Event
    cuda_stream = torch.cuda.Stream
    cuda_graph_pool_handle = torch.cuda.graph_pool_handle
    torch_event = getattr(torch, "Event", None)

    # Save original torch.zeros and torch.empty
    original_zeros = torch.zeros
    original_empty = torch.empty

    def _patched_zeros(*args, **kwargs):
        # Strip pin_memory on MPS - it's not supported
        kwargs.pop("pin_memory", None)
        return original_zeros(*args, **kwargs)

    def _patched_empty(*args, **kwargs):
        # Strip pin_memory on MPS - it's not supported
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    # Patch is_uva_available and is_pin_memory_available
    import vllm.utils.platform_utils as platform_utils

    original_is_uva_available = platform_utils.is_uva_available
    original_is_pin_memory_available = platform_utils.is_pin_memory_available

    # Patch UvaBuffer from vllm.v1.worker.gpu.states
    import vllm.v1.worker.gpu.states as states_module

    original_uva_buffer = states_module.UvaBuffer
    # Note: bincount is patched at module level, not here

    # Patch get_cuda_view_from_cpu_tensor
    import vllm.utils.torch_utils as torch_utils_module

    original_get_cuda_view = torch_utils_module.get_cuda_view_from_cpu_tensor

    def _patched_get_cuda_view(cpu_tensor: torch.Tensor) -> torch.Tensor:
        # For Metal, just return an MPS view of the tensor
        return cpu_tensor.to("mps")

    # Note: bincount is patched at module level via _patched_bincount_metal

    # Clear the cache to ensure our patched function is used
    if hasattr(platform_utils.is_uva_available, "cache_clear"):
        platform_utils.is_uva_available.cache_clear()
    if hasattr(platform_utils.is_pin_memory_available, "cache_clear"):
        platform_utils.is_pin_memory_available.cache_clear()

    try:
        # Replace with placeholders
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.graph_pool_handle = lambda: None  # type: ignore[return-value]
        if torch_event is not None:
            torch.Event = _EventPlaceholder

        # Patch torch.zeros and torch.empty to strip pin_memory
        torch.zeros = _patched_zeros
        torch.empty = _patched_empty

        # Metal unified memory acts like UVA
        platform_utils.is_uva_available = lambda: True
        platform_utils.is_pin_memory_available = lambda: True

        # Replace UvaBuffer with Metal-compatible version
        states_module.UvaBuffer = _MetalUvaBuffer

        # Replace get_cuda_view_from_cpu_tensor
        torch_utils_module.get_cuda_view_from_cpu_tensor = _patched_get_cuda_view

        # bincount is already patched at module level

        yield
    finally:
        # Restore original classes
        torch.cuda.Event = cuda_event
        torch.cuda.Stream = cuda_stream
        torch.cuda.graph_pool_handle = cuda_graph_pool_handle
        if torch_event is not None:
            torch.Event = torch_event

        # Restore original torch.zeros and torch.empty
        torch.zeros = original_zeros
        torch.empty = original_empty

        # Restore original functions
        platform_utils.is_uva_available = original_is_uva_available
        platform_utils.is_pin_memory_available = original_is_pin_memory_available

        # Restore UvaBuffer
        states_module.UvaBuffer = original_uva_buffer

        # Restore get_cuda_view_from_cpu_tensor
        torch_utils_module.get_cuda_view_from_cpu_tensor = original_get_cuda_view

        # Note: bincount stays patched (patched at module level)

        # Clear the cache again
        if hasattr(platform_utils.is_uva_available, "cache_clear"):
            platform_utils.is_uva_available.cache_clear()
        if hasattr(platform_utils.is_pin_memory_available, "cache_clear"):
            platform_utils.is_pin_memory_available.cache_clear()
