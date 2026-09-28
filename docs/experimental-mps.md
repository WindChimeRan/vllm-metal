# Experimental PyTorch MPS backend

This experimental backend runs upstream Model Runner V2 and its sampler on
PyTorch MPS. Qwen3 loads through vLLM's Transformers backend with hardware-agnostic
RMSNorm and SiLU implementations. Attention dispatches the existing paged/varlen
Metal kernels on PyTorch's stream.
[Upstream model integration PR](https://github.com/vllm-project/vllm/pull/49458).

Hardware-agnostic dispatch currently covers norms and activation. Projections,
embeddings, logits processing and the attention wrapper use the shared vLLM
layers; this does not load a hardware-specific flat Qwen3 implementation.

The current scope is single-device, unquantized Qwen3 with 128-dimensional heads
and 16-token cache blocks. Qwen3-0.6B FP16 has been exercised end to end;
attention tests also cover BF16. Chunked prefill, prefix reuse and preemption use
upstream MRV2 request state. Execution is eager with synchronous scheduling;
logical compute/copy streams share the ordered MPS default stream. Input staging
uses explicit copies. The default backend remains MLX.

MPS selects `VLLM_USE_V2_MODEL_RUNNER=1` and `VLLM_USE_HW_AGNOSTIC=1` by default.
The MLX path still uses the legacy runner contract. Quantization, LoRA,
speculative decoding, multimodal input, prompt logprobs and structured output
remain outside the MPS scope.

```bash
PYTHONPATH=. VLLM_METAL_BACKEND=mps vllm serve Qwen/Qwen3-0.6B \
  --model-impl transformers --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.3
```

Use the repository's vLLM 0.30 environment with PyTorch 2.13. The native host
launcher requires Apple's command-line tools and Ninja (the `mps` extra). MLX
is still needed by shared startup and shader loading. Run the focused tests with
`PYTHONPATH=. VLLM_METAL_BACKEND=mps python -m pytest tests/test_mps_*.py`.

## Adapter boundaries

| Component | Upstream owns | Metal adapts |
| --- | --- | --- |
| Runner | Model loading, scheduling handoff, persistent requests, preemption | Worker lifecycle and device configuration |
| Model | Transformers Qwen3, vLLM layer replacement and auxiliary hidden states | Attention backend and metadata builder |
| Sampling | MRV2 sampler, parameter state, filtering order and output assembly | RNG, penalties, constraints and logprob tensor operations |
| State | Request slots, block ownership and staged updates | MPS staging, input preparation and block/slot gathers |
| Device | PyTorch streams/events and MRV2 completion handling | Four CUDA-name bindings to native PyTorch MPS APIs |

The adapters register explicit bindings to vLLM's private 0.30 interfaces in the
MPS worker. `runtime.py` lists the device bindings; `input_ops.py` and `sampling_ops.py`
provide PyTorch replacements for device kernels. No upstream runner or sampler
control-flow method is copied. Metal-specific MRV2 validation retains upstream
feature checks while replacing the Triton requirement for this supported scope.
These bindings need upstream extension points before claiming compatibility with
later vLLM releases. There are no custom stream or event classes; synchronization
uses native completion events and device-level synchronization.

## Roadmap

- **MLX runner compatibility:** Migrate the MLX path to MRV2 contracts before the
  targeted MRV1 removal in v0.32.
  [Release announcement](https://github.com/vllm-project/vllm/releases/tag/v0.29.0).
- **Memory and warmup:** Profile loading, prefill, decode and sampling peaks;
  warm representative shapes. Budget weights, KV cache, scratch and driver
  allocations within unified memory, with reclamation and system headroom.
- **Shared buffers:** Reuse CPU-writable MPS input buffers and GPU workspaces.
  Define ownership, lifetimes and completion fences for zero-copy access and
  asynchronous execution; reduce staging copies and temporary allocations.
- **KV storage:** Handle Metal buffer-size limits, block copying and zeroing,
  prefix reuse, preemption and allocation failure across large caches.
- **Quantization and coverage:** Integrate existing
  [PyTorch MPS INT4/INT8 kernels](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/mps/operations/Quantized.mm#L72)
  with vLLM weight loading. Validate packed formats and quantized dense models,
  then hybrid/MoE execution, additional attention variants, LoRA and parallelism.
- **Hybrid models, separate follow-ups:** Start with LFM2.5's short-convolution
  state, then Qwen3.5's Gated DeltaNet state. Reuse upstream model and MRV2 state
  handling; adapt unsupported operations and validate prefix reuse, chunking and
  preemption for each state family. Keep vision and MoE support separate.
- **Execution optimization:** Reduce synchronization and allocation overhead;
  evaluate asynchronous pipelining and compilation on representative workloads.
- **Speculative decoding:** Collaborate upstream on shared hidden-state
  extraction, drafting, verification and cache/RNG contracts.
  [MRV2 hidden-state extraction PR](https://github.com/vllm-project/vllm/pull/49811).
- **Omni:** Collaborate on multimodal encoders and stage execution, including
  tensor ownership, streaming and memory budgets across stages.
- **Dependencies and kernels:** Decouple startup and shader loading from MLX;
  define supported dependency versions and upstream MPS fixes. Reuse existing
  Metal kernels; keep new kernels small and use TileLang.
- **Broader validation and rollout:** Extend coverage across model families,
  quantization formats and GPU generations under sustained load and memory
  pressure. Evaluate a default-backend change once coverage and performance
  justify it.
