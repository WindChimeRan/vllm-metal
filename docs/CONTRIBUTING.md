# Contributing to vLLM Metal

To run a released build, use the [installation guide](installation.md).
The setup below is for editing vllm-metal itself.

## Development setup

Fork the repository on GitHub, then clone your fork (replace `YOUR_USERNAME`):

```bash
git clone https://github.com/YOUR_USERNAME/vllm-metal.git
cd vllm-metal
git remote add upstream https://github.com/vllm-project/vllm-metal.git
git switch -c my-change
./install.sh --editable
source .venv-vllm-metal/bin/activate
```

Python changes take effect after restarting the process. This installs
editable Python sources and matching prebuilt native binaries without a compiler.
The native sources and MLX/nanobind versions must match the release wheel;
otherwise the installer asks you to build from source.

## Editing the Metal kernels

Kernel contributors build the native extension and shaders explicitly:

```bash
./install.sh --build
source .venv-vllm-metal/bin/activate
export VLLM_METAL_BUILD_FROM_SOURCE=1
```

`--build` needs Xcode with macOS SDK 26.2 or newer, `clang++`, and the Metal
compiler component. If the component is missing, install it with
`xcodebuild -downloadComponent MetalToolchain`. The installer reports compiler
errors without downloading tools automatically.

In source mode the C++ extension rebuilds when its inputs change and MLX
compiles changed shaders in-process. Restart after each kernel edit. To refresh
the prebuilt artifacts instead, run `python -m vllm_metal.metal.build`.
Stale local artifacts are rejected when source mode is disabled.

## Checks

Run from the repository root:

```bash
scripts/lint.sh
scripts/test.sh
```

For a shorter loop, run `pytest -m "not slow" tests/` in the activated environment.
Model parity runs separately through [scheduled and requested CI](tools.md#scheduled-and-requested-ci).

## Pull requests

- **Model changes:** run the [greedy parity tool](tools.md) against the environment's native `mlx-lm`. Report `EXACT` and `TOP_K_MATCH` counts separately and investigate failures.
- **Performance claims:** include before/after [serving benchmark](https://docs.vllm.ai/en/latest/cli/bench/serve/) results.

Sign off each commit to certify agreement with the [Developer Certificate of
Origin](https://developercertificate.org/), then push to your fork:

```bash
git commit -s -m "Describe your change"
git push -u origin my-change
```

Open a pull request against `main` in `vllm-project/vllm-metal`.

## Building documentation

```bash
uv pip install -r docs/requirements-docs.txt
mkdocs serve
```
