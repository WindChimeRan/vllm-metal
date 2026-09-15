# Contributing to vLLM Metal

Thanks for your interest in contributing! This plugin targets **Apple Silicon Macs only** — you'll need an M-series Mac running macOS to build, test, and run it.

## Development setup

```bash
git clone https://github.com/vllm-project/vllm-metal.git
cd vllm-metal

./install.sh --editable
source .venv-vllm-metal/bin/activate
```

Python changes take effect immediately. This installs development dependencies
and reuses matching native binaries from the latest release, without a compiler.
The native sources and MLX/nanobind versions must match the wheel; otherwise the
installer reports the mismatch and asks you to build from source.

## Editing the Metal kernels

Kernel contributors build the native extension and shaders explicitly:

```bash
./install.sh --build
source .venv-vllm-metal/bin/activate

# Recompile changed kernels on subsequent runs.
export VLLM_METAL_BUILD_FROM_SOURCE=1
```

In this mode the C++ extension is recompiled when its inputs change (the build
is hash-checked, so unchanged sources are skipped) and the shaders are compiled
in-process by MLX from the `.metal` source at runtime. There is **no manual
`.metallib` rebuild step**: edit a kernel, restart the Python process, and the
change is picked up.

`--build` needs Xcode with macOS SDK 26.2 or newer, `clang++`, and the Metal
compiler component. Install the component with
`xcodebuild -downloadComponent MetalToolchain` if it is missing. The installer
reports compiler failures without downloading tools automatically.

After setup, `VLLM_METAL_BUILD_FROM_SOURCE=1` uses `clang++` for the C++ extension
and MLX for runtime shader compilation.

Without `VLLM_METAL_BUILD_FROM_SOURCE`, the prebuilt artifacts are loaded as-is.
If you edited a kernel source after building them locally, loading **fails
loudly** on the stale-hash mismatch rather than silently running the old kernel —
set the variable, or rerun `python -m vllm_metal.metal.build` to refresh the
prebuilt artifacts. (A plain wheel install ships no hash stamps, so end users
never hit this.)

## Run lint locally

Mirrors the `lint` job in CI (`ruff`, `ruff format --check`, `mypy`, `shellcheck`):

```bash
scripts/lint.sh
```

## Run CI locally

Mirrors the `test` job in CI: wheel validation, Metal platform checks, and the non-slow pytest suite. Model parity runs separately in the [daily and requested workflow](tools.md#scheduled-and-requested-ci):

This needs Xcode with macOS SDK 26.2 or newer and its Metal
compiler component (`xcodebuild -downloadComponent MetalToolchain`). CI installs
the component explicitly.

```bash
scripts/test.sh
```

> For a faster inner loop while iterating, run pytest directly:
>
> ```bash
> pytest -m "not slow" tests/ -v --tb=short
> ```

🎉 **Congratulations!** You have completed the development environment setup.

---

## Before you open the PR

Two conditional checks apply depending on what your PR touches:

**If your PR adds or modifies a model**, run the [greedy parity tool](tools.md) against the environment's native `mlx-lm`. Report `EXACT` and `TOP_K_MATCH` counts separately and investigate failures.

**If your PR claims a performance improvement**, attach before/after benchmark results. For example, using `vllm bench serve` with the sonnet dataset:

```bash
curl -O https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt

# 1. Start the server
VLLM_METAL_MEMORY_FRACTION=0.8 \
  vllm serve Qwen/Qwen3-0.6B --port 8000 --max-model-len 2048

# 2. Run the benchmark
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-0.6B \
  --dataset-name sonnet \
  --dataset-path sonnet.txt \
  --num-prompts 100 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,99
```

## Developer Certificate of Origin (DCO)

When contributing changes to this project, you must agree to the [DCO](https://developercertificate.org/). Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## Submit your changes

1. **Fork** the repository on GitHub.

2. **Re-point `origin` to your fork and add `upstream`:**

   ```bash
   git remote set-url origin https://github.com/<your-username>/vllm-metal.git
   git remote add upstream https://github.com/vllm-project/vllm-metal.git
   ```

3. **Create a feature branch:**

   ```bash
   git checkout -b my-feature
   ```

4. **Commit your changes using `-s`** (adds the DCO sign-off automatically):

   ```bash
   git commit -sm "your commit info"
   ```

5. **Push to your fork:**

   ```bash
   git push -u origin my-feature
   ```

6. **Open a pull request** against `main` in the upstream repository.
