# Speculative Decoding

Use vLLM's [speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/) for method behavior and configuration. This page documents only vllm-metal support and differences.

## Supported methods

| Method | Metal support | Upstream guide |
|---|---|---|
| Gemma4 MTP | Gemma4 target with a matching assistant checkpoint | [MTP](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) |
| Draft model | Non-hybrid paged-attention target with a full-attention draft model | [Draft model](https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model/) |
| N-gram | Non-hybrid paged-attention target | [N-gram](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/) |

All three methods currently have these Metal-specific constraints:

- Target verification requires paged attention.
- Only greedy requests (`temperature=0`) are drafted. Other requests run without speculation.
- Scheduling must be synchronous. The Metal platform disables async scheduling when speculative decoding is configured.
- Pipeline parallelism is not supported with speculative decoding.
- Hybrid GDN targets and heterogeneous draft vocabularies are not supported.
- `long_prefill_token_threshold`, when set, must be at least `1 + num_speculative_tokens`.

Unsupported combinations fail instead of falling back silently.

## Gemma4 MTP

Set `method` to `mtp` and pass the assistant checkpoint through `model`, as shown in the upstream [Gemma4 assistant guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/#gemma-4-assistant-models).

Use matching target and assistant families:

| Target | Assistant |
|---|---|
| Gemma4 E2B-it | Gemma4 E2B-it assistant bf16 |
| Gemma4 E4B-it | Gemma4 E4B-it assistant bf16 |
| Gemma4 31B-it bf16 | Gemma4 31B-it assistant bf16 |

Start with `num_speculative_tokens=3`. On the measured E4B workload, higher values improved single-stream throughput but reduced saturated throughput. Benchmark the intended batch shape before changing it.

Remote Hugging Face checkpoints are supported. Pin `revision` in `speculative_config` when publishing benchmark results.

## Draft model

The draft model must use full attention and the target vocabulary. Sliding-window and hybrid draft models are rejected at startup. Its committed KV cache is scheduler-managed and shares the Metal KV memory budget with the target.

## N-gram

N-gram speculation uses vLLM's prompt-lookup proposer and needs no additional model or KV cache. Its benefit depends on repeated token spans in the request history.

## Benchmarking

Use vLLM's benchmark CLI for serving workloads. For a reproducible Gemma4 target-only versus MTP comparison, use the in-tree benchmark:

```bash
python -m tools.benchmark.gemma4_mtp_benchmark --help
```

`tools/README.md` documents the before-and-after commands and the natural-prompt dataset used for speculative-decoding measurements.
