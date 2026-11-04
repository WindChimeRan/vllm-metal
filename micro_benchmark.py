"""Synchronized sampling draw latency, excluding model execution and filtering.

MPS uses vLLM's public gumbel_sample wrapper for both implementations.
The MLX baseline is the per-row seeded categorical draw from the PR base a44be487;
its filter stage and seed derivation are outside the timed region.
"""
import argparse
import importlib.util
from contextlib import nullcontext
import json
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=('mps', 'mlx'), required=True)
parser.add_argument('--variant', choices=('baseline', 'shared'), required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--calls', type=int, default=200)
parser.add_argument('--warmups', type=int, default=10)
parser.add_argument('--warmup-seconds', type=float, default=0.5)
args = parser.parse_args()

if args.backend == 'mps':
    import torch
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from vllm_metal.pytorch_backend.bindings import register_kernel_overrides
    from vllm_metal.pytorch_backend import sampling
    register_kernel_overrides()
    if args.variant == 'baseline':
        from vllm.triton_utils.dispatcher import register_kernels
        spec = importlib.util.spec_from_file_location('pytorch_reference', Path(__file__).with_name('pytorch_reference.py'))
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        register_kernels({'vllm.v1.worker.gpu.sample.gumbel._gumbel_sample_kernel': reference.gumbel_sample})
    else:
        assert hasattr(sampling, '_gumbel_kernel')
    torch.set_num_threads(4)
else:
    import mlx.core as mx
    from vllm_metal.metal import get_ops
    # Resolve the shared adapter before timing or warming any case.
    ops = get_ops() if args.variant == 'shared' else None

context = torch.inference_mode() if args.backend == 'mps' else nullcontext()
with context:
    results = []
    steps = args.calls + args.warmups
    for vocab in (6562, 152064):
        for batch in (1, 4, 8):
            # Fresh logits and RNG inputs for every call prevent result reuse.
            rng = np.random.default_rng(20261102 + vocab + batch)
            host = rng.normal(size=(steps, batch, vocab)).astype(np.float32)
            host /= np.float32(.8)
            if args.backend == 'mps':
                logits = torch.from_numpy(host).to('mps')
                mapping = torch.arange(batch, dtype=torch.int32, device='mps')
                seeds = torch.arange(batch, device='mps') * 731 + 42
                temperatures = torch.ones(batch, device='mps')
                positions = [torch.full((batch,), i, dtype=torch.int64, device='mps') for i in range(steps)]
                def draw(i):
                    return gumbel_sample(logits[i], mapping, temperatures, seeds, positions[i], False, False)
                synchronize = torch.mps.synchronize
                evaluate = lambda result: synchronize()
            else:
                logits = mx.array(host)
                row_keys = [[mx.random.key(42 + i * 1009 + row * 731) for row in range(batch)] for i in range(steps)]
                positions = mx.zeros((batch,), dtype=mx.int64)
                temperatures = mx.ones((batch,), dtype=mx.float32)
                if args.variant == 'baseline':
                    def draw(i):
                        return mx.stack([mx.random.categorical(logits[i, row], key=row_keys[i][row]) for row in range(batch)])
                else:
                    def draw(i):
                        keys = mx.stack(row_keys[i]).view(mx.int64).reshape(-1)
                        return ops.gumbel_sample(logits[i], keys, positions, temperatures)
                synchronize = mx.synchronize
                evaluate = mx.eval
                mx.eval(logits, positions, temperatures, *[key for keys in row_keys for key in keys])
            del host
            warmup_start = time.perf_counter()
            warmup_calls = 0
            while warmup_calls < args.warmups or time.perf_counter() - warmup_start < args.warmup_seconds:
                evaluate(draw(warmup_calls % args.warmups))
                warmup_calls += 1
            synchronize()
            sampled = []
            start = time.perf_counter_ns()
            for i in range(args.warmups, steps):
                result = draw(i)
                evaluate(result)
                sampled.append(result)
            elapsed_ns = time.perf_counter_ns() - start
            all_tokens = torch.stack(sampled).cpu().tolist() if args.backend == 'mps' else mx.stack(sampled).tolist()
            tokens = result.cpu().tolist() if args.backend == 'mps' else result.tolist()
            assert len(tokens) == batch and all(0 <= t < vocab for t in tokens)
            row = {'backend': args.backend, 'variant': args.variant, 'batch': batch,
                   'vocab': vocab, 'warmups': warmup_calls, 'minimum_warmup_seconds': args.warmup_seconds, 'calls': args.calls,
                   'microseconds_per_draw': elapsed_ns / args.calls / 1000,
                   'last_tokens': tokens, 'all_tokens': all_tokens}
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2))
            print(json.dumps(row), flush=True)
            del result, logits, draw, sampled
            synchronize()
print('CORRECT=True (finite token IDs; MPS baseline equality checked by orchestrator)', flush=True)
