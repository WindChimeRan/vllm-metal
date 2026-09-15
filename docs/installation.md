# Installation

## Requirements

- macOS 15 (Sequoia) or later, on Apple Silicon. The prebuilt wheel's
  native extension and Metal kernels target macOS 15, so the wheel is
  tagged `macosx_15_0_arm64` and will not install on earlier releases.
- Native arm64 Python 3.12. Rosetta/x86_64 Python is not supported.

> **No compiler required.** The install script below fetches vLLM core and the
> vllm-metal plugin as prebuilt wheels, so nothing is compiled on your machine.
> Running `./install.sh` from a checkout installs the same release wheels.
> Python contributors use `./install.sh --editable`; kernel contributors use
> `./install.sh --build`. See [Contributing](CONTRIBUTING.md).

`uv` is bootstrapped automatically.

## Install

Using the install script, the following will be installed under the `~/.venv-vllm-metal` directory (the default).
- vllm-metal plugin
- vllm core
- Related libraries

If you run `source ~/.venv-vllm-metal/bin/activate`, the `vllm` CLI becomes available and you can access the vLLM right away.

For how to use the `vllm` CLI, please refer to the [official vLLM guide](https://docs.vllm.ai/en/latest/cli/).

Development channel (default):

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

Stable channel:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash -s -- --stable
```

`pip install vllm-metal` is not supported. Use one of the commands above.

## Reinstallation and Update

If any issues occur, please use the following command to switch to the latest release version and check if the problem is resolved.
If the issue continues to occur in the latest release, please report the details of the issue.
(If you have installed it in a directory other than the default `~/.venv-vllm-metal`, substitute that path and run the command accordingly.)

```bash
rm -rf ~/.venv-vllm-metal && curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## Uninstall

Please delete the directory that was installed by the installation script.
(If you have installed it in a directory other than the default `~/.venv-vllm-metal`, substitute that path and run the command accordingly.)

```bash
rm -rf ~/.venv-vllm-metal
```

## Building Documentation

```bash
uv pip install -r docs/requirements-docs.txt
mkdocs serve
```
