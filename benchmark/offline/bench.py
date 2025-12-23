# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py
# Adapted from: https://github.com/sgl-project/mini-sglang/blob/main/benchmark/offline/bench.py
# Adapted for vllm-metal

import time
from random import randint, seed

from mlx_lm import stream_generate

from vllm_metal.model_runner import MetalModelRunner


class MinimalConfig:
    """Minimal config for MetalModelRunner."""

    class ModelConfig:
        def __init__(self, model: str):
            self.model = model
            self.trust_remote_code = True

    def __init__(self, model: str):
        self.model_config = self.ModelConfig(model)


def main():
    seed(0)
    # num_seqs = 256
    # max_input_len = 1024
    # max_output_len = 1024
    num_seqs = 4
    max_input_len = 128
    max_output_len = 128

    # Use MetalModelRunner to load the model
    runner = MetalModelRunner(MinimalConfig("Qwen/Qwen3-0.6B"))
    runner.load_model()

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    output_lens = [randint(100, max_output_len) for _ in range(num_seqs)]

    # Warmup
    runner.generate("Warmup", max_tokens=10)

    # Benchmark using stream_generate with token IDs
    t = time.time()
    for tokens, max_tokens in zip(prompt_token_ids, output_lens):
        for _ in stream_generate(
            runner.model, runner.tokenizer, tokens, max_tokens=max_tokens
        ):
            pass
    t = time.time() - t

    total_tokens = sum(output_lens)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
