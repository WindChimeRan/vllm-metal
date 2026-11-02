# Sampling benchmark evidence for vllm-metal #1234

The figures and numeric results are review assets, separate from the implementation PR.

- End-to-end: Qwen3-0.6B FP16, 32 requests, 128 input/128 output tokens, C=1/4/8, two warmups, one measured run per setting. MLX baseline is main 3136697d including #1233; MPS baseline is the prior PyTorch-operator sampler 09fbdaad. Shared implementation: 5d136d81. All 960 requests completed.
- Microbenchmark: B=1/4/8, vocabulary 6,562 and 152,064, fresh FP32 logits already scaled by temperature 0.8. Ten warmups and one window of 200 draws, synchronized after every draw. MPS uses torch.inference_mode. Includes host dispatch, allocation, GPU work and reduction; excludes model execution, filtering and seed derivation. This is not device-only shader timing.
- Micro baselines: frozen PyTorch-operator Gumbel sampler; MLX per-row seeded categorical draw as used by #1233. The shared adapters use the same Metal shader. MPS reference tokens matched for every checked shape; MLX RNG streams differ by design.
- Hardware/software: Apple M5 Pro, 64 GiB; PyTorch 2.16.0.dev20261013, MLX 0.32.4, vLLM 0.32.0+cpu. MPS CPU fallback disabled.
- Speech-sized micro inputs do not establish an MLX Omni serving integration. Existing Omni serving uses PyTorch-MPS.

Run micro_benchmark.py from an environment with the PR installed, with --backend mps or mlx, --variant baseline or shared, and --output result.json. Keep pytorch_reference.py beside it. Set PYTORCH_ENABLE_MPS_FALLBACK=0.
