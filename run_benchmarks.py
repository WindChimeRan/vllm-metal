"""Refresh the serving and draw comparisons for PR #1234."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = Path("/Users/ranzhang/.codex/worktrees/minicpmo-duplex/vllm-metal")
CANDIDATE = OUTPUT.parent / "pr1234-two-levels/code"
BASELINE = Path("/private/tmp/metal-1234-overlap-tQNvETfJ/plain")
TORCH_REFERENCE = ROOT / ".cache/pr-metal-sampling"
OMNI = ROOT / ".cache/pr-omni-pinned"
MODEL = "/Users/ranzhang/workspace/vllm-metal/.cache/spec-parity-models/Qwen3-0.6B-fp16"
VARIANTS = (
    ("mlx-baseline", "mlx", BASELINE),
    ("mlx-shared", "mlx", CANDIDATE),
    ("mps-baseline", "mps", TORCH_REFERENCE),
    ("mps-shared", "mps", CANDIDATE),
)


def environment(repo, backend):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_", "PYTORCH_", "MLX_"))}
    env.update(
        PYTHONPATH=str(repo) + ":" + str(OMNI),
        PYTHONDONTWRITEBYTECODE="1",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        VLLM_METAL_BACKEND=backend,
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        OMP_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        VLLM_NO_USAGE_STATS="1",
        NO_PROXY="127.0.0.1,localhost",
    )
    return env


def run_logged(command, path, repo, env, timeout=300):
    with path.open("w") as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        print(path.read_text()[-8000:], flush=True)
        raise RuntimeError(f"Command exited {result.returncode}: {path}")


def serving():
    rows = []
    for name, backend, repo in VARIANTS:
        out = OUTPUT / "serving" / name
        out.mkdir(parents=True, exist_ok=False)
        env = environment(repo, backend)
        with socket.socket() as socket_probe:
            socket_probe.bind(("127.0.0.1", 0))
            port = str(socket_probe.getsockname()[1])
        base_url = "http://127.0.0.1:" + port
        command = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
            "--served-model-name", "bench", "--host", "127.0.0.1", "--port", port,
            "--max-model-len", "2048", "--max-num-seqs", "8", "--max-num-batched-tokens", "512",
            "--gpu-memory-utilization", ".3", "--no-enable-prefix-caching",
            "--generation-config", "vllm", "--enforce-eager",
        ]
        metadata = dict(repo=str(repo), head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            diff=subprocess.check_output(["git", "diff"], cwd=repo, text=True), command=command)
        (out / "command.json").write_text(json.dumps(metadata, indent=2))
        with (out / "server.log").open("w") as log:
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                for _ in range(240):
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited: {out / 'server.log'}")
                    try:
                        with urllib.request.urlopen(base_url + "/health", timeout=1):
                            break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(1)
                else:
                    raise TimeoutError(name + " startup")
                print(name, "ready", flush=True)
                cases = [
                    ("plain-seeded", {"seed": 42, "top_k": -1, "top_p": 1.0}),
                    ("filtered-seeded", {"seed": 42, "top_k": 50, "top_p": .9}),
                ]
                if backend == "mlx":
                    cases.append(("filtered-unseeded", {"top_k": 50, "top_p": .9}))
                for case, extra in cases:
                    for concurrency in (1, 4, 8):
                        tag = f"{case}-c{concurrency}"
                        bench = [
                            sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
                            "--backend", "openai", "--base-url", base_url, "--endpoint", "/v1/completions",
                            "--model", MODEL, "--served-model-name", "bench", "--dataset-name", "random",
                            "--random-input-len", "128", "--random-output-len", "128", "--random-range-ratio", "0",
                            "--num-prompts", "32", "--num-warmups", "8", "--request-rate", "inf",
                            "--max-concurrency", str(concurrency), "--ignore-eos", "--temperature", ".8",
                            "--seed", "42", "--extra-body", json.dumps(extra), "--disable-tqdm",
                            "--save-result", "--save-detailed", "--result-dir", str(out),
                            "--result-filename", tag + ".json",
                        ]
                        (out / (tag + "-command.json")).write_text(json.dumps(bench, indent=2))
                        run_logged(bench, out / (tag + ".log"), repo, env)
                        result = json.loads((out / (tag + ".json")).read_text())
                        assert result["completed"] == 32 and result["total_output_tokens"] == 4096
                        row = dict(variant=name, backend=backend, case=case, concurrency=concurrency,
                                   **{k: result.get(k) for k in (
                                       "completed", "failed", "total_input_tokens", "total_output_tokens",
                                       "output_throughput", "mean_tpot_ms", "median_tpot_ms",
                                       "p99_tpot_ms", "mean_ttft_ms", "mean_itl_ms")})
                        rows.append(row)
                        (OUTPUT / "serving-results.json").write_text(json.dumps(rows, indent=2))
                        print(name, case, concurrency, round(row["mean_tpot_ms"], 3), "ms/token", flush=True)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def microbenchmark():
    out = OUTPUT / "micro"
    out.mkdir(exist_ok=False)
    for name, backend, repo in VARIANTS:
        variant = name.split("-")[1]
        # Both MPS draws use the current public wrapper and override registration.
        if backend == "mps":
            repo = CANDIDATE
        command = [
            sys.executable, str(OUTPUT / "micro_benchmark.py"), "--backend", backend,
            "--variant", variant, "--output", str(out / (name + ".json")),
        ]
        run_logged(command, out / (name + ".log"), repo, environment(repo, backend))
        print(name, "draw benchmark complete", flush=True)
    rows = []
    for backend in ("mlx", "mps"):
        before = json.loads((out / f"{backend}-baseline.json").read_text())
        after = json.loads((out / f"{backend}-shared.json").read_text())
        for a, b in zip(before, after, strict=True):
            assert (a["batch"], a["vocab"]) == (b["batch"], b["vocab"])
            if backend == "mps":
                assert a["all_tokens"] == b["all_tokens"], (a["batch"], a["vocab"])
            rows.append(dict(backend=backend, batch=a["batch"], vocab=a["vocab"],
                             baseline_us=a["microseconds_per_draw"], shared_us=b["microseconds_per_draw"]))
    (OUTPUT / "micro-results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "serving":
        serving()
    if len(sys.argv) == 1 or sys.argv[1] == "micro":
        microbenchmark()
