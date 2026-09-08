# SPDX-License-Identifier: Apache-2.0
"""One-engine-per-process EAGLE3 correctness and throughput measurements.

Run off, draft_model, and eagle3 separately, retaining exact output token IDs.
Every run uses greedy sampling, prefix caching, and synchronous scheduling.
The cold and reused-prefix phases are reported separately, never averaged.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

from tools.benchmark.gemma4_mtp_benchmark import environment_metadata
from tools.check_parity import compare_results

TARGETS = {
    "qwen": ("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B", "RedHatAI/Qwen3-8B-speculator.eagle3"),
    "llama": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    ),
}

QUESTIONS = [
    "Explain how an LRU cache works and give a small example showing which entry is evicted. Use clear, concrete language.",
    "Write a Python function that removes duplicate elements from a list while preserving their original order. Explain its time complexity.",
    "A library starts with 120 books. It receives 35 donated books, lends 48 books, and gets 19 books back. How many books are now in the library? Explain the calculation.",
    "Compare solar panels and wind turbines for generating electricity. Describe one strength and one limitation of each, then explain how a grid can use both.",
]
SYSTEM = (
    "You are a helpful assistant. Answer the user's question accurately and directly. "
    "Use complete sentences and concrete examples where they help. Avoid unnecessary "
    "introductory remarks. For programming questions, provide working code and briefly "
    "explain the approach. For arithmetic, show the calculation and state the answer."
)


@contextmanager
def metal_trace(llm, top_k):
    """Tool-only observation of genuine greedy rows; never enable logprobs."""
    import mlx.core as mx
    import numpy as np

    import vllm_metal.v1.model_runner as runner_module
    import vllm_metal.v1.sampling_batch as sampling

    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    controller = runner._spec_decode_controller
    tokenizer = llm.get_tokenizer()
    scores = {}
    counts = {"rounds": 0, "drafted": 0, "accepted": 0}
    sample = sampling.sample_from_logits
    verify = controller.verify_greedy

    def record(logits, prompt, previous):
        if top_k is None:
            return
        row = logits.astype(mx.float32)
        logprobs = np.array(row - mx.logsumexp(row))
        indices = np.argsort(-logprobs, kind="stable")[:top_k]
        scores[tuple(prompt), tuple(previous)] = [
            {
                "id": int(i),
                "text": tokenizer.decode([int(i)]),
                "rank": rank,
                "logprob": float(logprobs[i]),
            }
            for rank, i in enumerate(indices, start=1)
        ]

    def observed_sample(logits, batch, sampler):
        result = sample(logits, batch, sampler)
        for i, (prompt, previous) in enumerate(
            zip(batch.prompt_token_id_lists, batch.output_token_id_lists, strict=True)
        ):
            record(logits[i], prompt, previous)
        return result

    def observed_verify(logits, reqs, segments):
        outputs = verify(logits, reqs, segments)
        for (_, state), segment, tokens in zip(reqs, segments, outputs, strict=True):
            counts["rounds"] += 1
            counts["drafted"] += len(segment.draft_token_ids)
            counts["accepted"] += len(tokens) - 1
            for i in range(len(tokens)):
                record(
                    logits[0, segment.start_row + i],
                    state.token_ids[: state.prompt_len],
                    state.token_ids[state.prompt_len :] + tokens[:i],
                )
        return outputs

    with (
        patch.object(sampling, "sample_from_logits", observed_sample),
        patch.object(runner_module, "sample_from_logits", observed_sample),
        patch.object(controller, "verify_greedy", observed_verify),
    ):
        yield scores, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument(
        "--drafter",
        help="Override the default draft checkpoint with a model ID or local path",
    )
    parser.add_argument(
        "--method", choices=["off", "draft_model", "eagle3"], required=True
    )
    parser.add_argument("--k", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument(
        "--dynamic-k",
        action="store_true",
        help="Use K for one active request and zero for larger batches",
    )
    parser.add_argument(
        "--self-draft",
        action="store_true",
        help="Use the target checkpoint as the ordinary draft for an acceptance sanity check",
    )
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--respect-eos", action="store_true")
    parser.add_argument(
        "--staggered-lengths",
        action="store_true",
        help="Use a long first request and shorter peers to exercise batch transitions",
    )
    parser.add_argument("--context-file", type=Path, default=Path("sonnet.txt"))
    parser.add_argument("--distinct-contexts", action="store_true")
    parser.add_argument("--cache-blocks", type=int)
    parser.add_argument("--require-preemption", action="store_true")
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help="Prepend this many tokens of the context file to each question",
    )
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--prefill-chunk", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--top-k",
        type=int,
        help="Diagnostic comparison using the shared parity tool; collect in both runs, never use these timings",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Target-only JSON; exact tokens unless --top-k is set",
    )
    args = parser.parse_args()
    if args.self_draft and args.drafter:
        parser.error("--self-draft cannot be combined with --drafter")
    if args.top_k is not None and (
        args.top_k < 1 or args.respect_eos or args.staggered_lengths
    ):
        parser.error("--top-k requires positive K and fixed output lengths")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if (
        args.top_k is not None
        and reference
        and (reference.get("top_k") or 0) < args.top_k
    ):
        parser.error("The reference must also be generated with at least this --top-k")
    reference_runs = (
        {(run["batch"], run["repeat"], run["phase"]): run for run in reference["runs"]}
        if reference
        else {}
    )
    any_mismatch = False
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    batches = [int(value) for value in args.batches.split(",")]
    if args.top_k is not None and max(batches) > len(QUESTIONS):
        parser.error("--top-k requires unique prompts; use batches of at most four")
    target, draft, eagle = TARGETS[args.target]
    if args.self_draft:
        if args.method != "draft_model":
            parser.error("--self-draft requires --method draft_model")
        draft = target
    # Resolve cached local paths to avoid network metadata probes during timing.
    target_path = snapshot_download(
        target, local_files_only=True, allow_patterns=["config.json"]
    )
    kwargs = {
        "model": target_path,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(batches),
        "max_num_batched_tokens": args.prefill_chunk,
        "enable_prefix_caching": True,
        "async_scheduling": False,
        "dtype": args.dtype,
        "disable_log_stats": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.cache_blocks is not None:
        kwargs["num_gpu_blocks_override"] = args.cache_blocks
    if args.method != "off":
        head = args.drafter or (eagle if args.method == "eagle3" else draft)
        head_path = (
            str(Path(head).resolve())
            if Path(head).is_dir()
            else snapshot_download(
                head, local_files_only=True, allow_patterns=["config.json"]
            )
        )
        kwargs["speculative_config"] = {
            "method": args.method,
            "model": head_path,
            "num_speculative_tokens": args.k,
        }
        if args.dynamic_k:
            schedule = [[1, 1, args.k]]
            if max(batches) > 1:
                schedule.append([2, max(batches), 0])
            kwargs["speculative_config"]["num_speculative_tokens_per_batch_size"] = (
                schedule
            )
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    contexts = [""] * len(QUESTIONS)
    if args.context_tokens:
        context_ids = tokenizer.encode(
            args.context_file.read_text(), add_special_tokens=False
        )
        needed = args.context_tokens * (len(QUESTIONS) if args.distinct_contexts else 1)
        if needed > len(context_ids):
            raise ValueError("Context file is shorter than --context-tokens")
        for i in range(len(QUESTIONS)):
            start = i * args.context_tokens if args.distinct_contexts else 0
            contexts[i] = (
                "Background reading:\n"
                + tokenizer.decode(context_ids[start : start + args.context_tokens])
                + "\n\nQuestion:\n"
            )
    prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": context + question},
                ],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                **({"enable_thinking": False} if args.target == "qwen" else {}),
            )
        }
        for context, question in zip(contexts, QUESTIONS, strict=True)
    ]
    sampling = SamplingParams(
        temperature=0, max_tokens=args.output_len, ignore_eos=not args.respect_eos
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    controller = runner._spec_decode_controller
    verify = controller.verify_greedy
    counters = {"rounds": 0, "drafted": 0, "accepted": 0}
    events = {"preemptions": 0}
    budgets = {}
    if args.dynamic_k and runner._drafter is not None:
        propose = runner._drafter.propose

        def observe_budget(ctx):
            key = str(ctx.num_speculative_tokens)
            budgets[key] = budgets.get(key, 0) + 1
            return propose(ctx)

        runner._drafter.propose = observe_budget
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    preempt = scheduler._preempt_request

    def observe_preemption(*args, **kwargs):
        events["preemptions"] += 1
        return preempt(*args, **kwargs)

    scheduler._preempt_request = observe_preemption

    def observe(logits, reqs, segments):
        result = verify(logits, reqs, segments)
        for segment, output in zip(segments, result, strict=True):
            counters["rounds"] += 1
            counters["drafted"] += len(segment.draft_token_ids)
            counters["accepted"] += len(output) - 1
        return result

    controller.verify_greedy = observe
    # Warm the engine independently of the measured prompt prefix.
    llm.generate(
        "Describe a bicycle.",
        SamplingParams(temperature=0, max_tokens=8),
        use_tqdm=False,
    )
    result = {
        "target": target,
        "method": args.method,
        "k": args.k,
        "config": kwargs,
        "environment": environment_metadata(),
        "device": mx.device_info(),
        "sampling": {
            "temperature": 0,
            "max_tokens": args.output_len,
            "ignore_eos": not args.respect_eos,
            "staggered_lengths": args.staggered_lengths,
        },
        "context_tokens": args.context_tokens,
        "top_k": args.top_k,
        "diagnostic_only": args.top_k is not None,
        "runs": [],
    }
    head = getattr(runner._forward_model, "lm_head", None)
    result["target_lm_head_dtype"] = (
        str(head.weight.dtype) if head is not None else None
    )
    result["kv_cache_dtype"] = str(runner.kv_cache_dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in batches:
        selected = [prompts[i % len(prompts)] for i in range(batch)]
        batch_sampling = (
            [
                SamplingParams(
                    temperature=0,
                    max_tokens=args.output_len
                    if i == 0
                    else min(args.output_len, 16 * (i + 1)),
                    ignore_eos=not args.respect_eos,
                )
                for i in range(batch)
            ]
            if args.staggered_lengths
            else sampling
        )
        for repeat in range(args.repeats):
            llm.reset_prefix_cache()
            for phase in ("cold", "cached"):
                for key in counters:
                    counters[key] = 0
                events["preemptions"] = 0
                budgets.clear()
                mx.reset_peak_memory()
                start = time.perf_counter()
                with (
                    metal_trace(llm, args.top_k)
                    if args.top_k is not None
                    else nullcontext(({}, {}))
                ) as (scores, _):
                    outputs = llm.generate(selected, batch_sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
                if (
                    args.method != "off"
                    and not args.dynamic_k
                    and args.output_len > 1
                    and not counters["drafted"]
                ):
                    raise RuntimeError(
                        "No draft tokens were verified; speculation was not tested"
                    )
                samples = [
                    {
                        "prompt_token_ids": o.prompt_token_ids,
                        "token_ids": list(o.outputs[0].token_ids),
                        "text": o.outputs[0].text,
                        "cached_tokens": o.num_cached_tokens,
                        "top_logprobs": [
                            scores[
                                tuple(o.prompt_token_ids),
                                tuple(o.outputs[0].token_ids[:i]),
                            ]
                            for i in range(len(o.outputs[0].token_ids))
                        ]
                        if args.top_k is not None
                        else [],
                    }
                    for o in outputs
                ]
                output_tokens = sum(len(s["token_ids"]) for s in samples)
                run = {
                    "batch": batch,
                    "repeat": repeat,
                    "phase": phase,
                    "elapsed_s": elapsed,
                    "output_tokens": output_tokens,
                    "output_tokens_per_s": output_tokens / elapsed,
                    "speculation": dict(counters),
                    "preemptions": events["preemptions"],
                    "draft_budget_counts": dict(budgets),
                    "mlx_peak_bytes": mx.get_peak_memory(),
                    "outputs": samples,
                }
                if reference is not None:
                    baseline = reference_runs[(batch, repeat, phase)]
                    mismatches = []
                    for index, (expected, actual) in enumerate(
                        zip(baseline["outputs"], samples, strict=True)
                    ):
                        if expected["prompt_token_ids"] != actual["prompt_token_ids"]:
                            raise ValueError(
                                "Reference and measured prompt token IDs differ"
                            )
                        if expected["token_ids"] != actual["token_ids"]:
                            first = next(
                                (
                                    i
                                    for i, (a, b) in enumerate(
                                        zip(
                                            expected["token_ids"],
                                            actual["token_ids"],
                                            strict=False,
                                        )
                                    )
                                    if a != b
                                ),
                                min(
                                    len(expected["token_ids"]), len(actual["token_ids"])
                                ),
                            )
                            mismatches.append({"request": index, "first_token": first})
                    run["token_mismatches"] = mismatches
                    if args.top_k is not None:

                        def parity_rows(rows):
                            return [
                                {
                                    "prompt": QUESTIONS[i],
                                    "tokens": row["token_ids"],
                                    "text": row["text"],
                                    "top_logprobs": row["top_logprobs"],
                                }
                                for i, row in enumerate(rows)
                            ]

                        passed = compare_results(
                            parity_rows(baseline["outputs"]),
                            parity_rows(samples),
                            max_tokens=args.output_len,
                            top_k=args.top_k,
                            reference_label="metal-off",
                        )
                        run["top_k_passed"] = passed
                        any_mismatch |= not passed
                    else:
                        any_mismatch |= bool(mismatches)
                result["runs"].append(run)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps({k: v for k, v in run.items() if k != "outputs"}),
                    flush=True,
                )
    if any_mismatch:
        raise SystemExit(
            "FAIL: speculative output failed the requested parity comparison"
        )
    if args.require_preemption and not any(
        run["preemptions"] for run in result["runs"]
    ):
        raise SystemExit("FAIL: requested pressure test did not trigger preemption")


if __name__ == "__main__":
    main()
