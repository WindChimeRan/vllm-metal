# Releasing

vllm-metal ships on two channels:

| Channel | Trigger | Version | Destination |
| --- | --- | --- | --- |
| **Dev** | every push to `main` (`release.yml`) | `0.3.0.devTIMESTAMP` | [GitHub Releases](https://github.com/vllm-project/vllm-metal/releases) |
| **Stable** | manual (`publish-pypi.yml`) | `[project].version` from `pyproject.toml` | [PyPI](https://pypi.org/project/vllm-metal/) |

This mirrors upstream vLLM, which publishes stable wheels to PyPI and per-commit
dev wheels to its own `wheels.vllm.ai` index. Both channels build the same wheel
via `build_wheel` in `scripts/lib.sh`, so the native `_paged_ops.so` + `.metallib`
shaders are always bundled and end users never compile.

## One-time setup (PyPI project owner)

Stable publishing uses **PyPI Trusted Publishing (OIDC)** — no API token is
stored in the repo. A maintainer with owner rights on the PyPI `vllm-metal`
project needs to do this once, on **both** PyPI and TestPyPI:

1. **Register the trusted publisher.** On the project's *Publishing* settings
   (e.g. <https://pypi.org/manage/project/vllm-metal/settings/publishing/>), add
   a new *GitHub* pending/existing publisher with:

   | Field | Value |
   | --- | --- |
   | Owner | `vllm-project` |
   | Repository | `vllm-metal` |
   | Workflow name | `publish-pypi.yml` |
   | Environment | `pypi` (on PyPI) / `testpypi` (on TestPyPI) |

   Repeat on <https://test.pypi.org/> with environment `testpypi`.

2. **Create the GitHub Environments** `pypi` and `testpypi`
   (repo *Settings → Environments*). Optionally add required reviewers on `pypi`
   so a human approves each production upload.

That's all the maintainer action needed — no secrets, nothing to rotate.

## Cutting a stable release

1. Bump `version` in `pyproject.toml` (the current PyPI release is stale at
   `0.1.0`; the first publish off this workflow is `0.3.0`). Land it on `main`.
2. **Rehearse on TestPyPI:** *Actions → Publish to PyPI → Run workflow*, target
   `testpypi`. Confirm the wheel uploads and
   `pip install -i https://test.pypi.org/simple/ vllm-metal` resolves the wheel.
3. **Publish:** run the workflow again with target `pypi`.

Each version can be uploaded only once; bump the version for any re-publish.

## Known limitation — `pip install vllm-metal` is not yet self-contained

Publishing the plugin to PyPI does **not** by itself make a bare
`pip install vllm-metal` a working install. vllm-metal deliberately does not
declare a hard `vllm` dependency, because **vLLM ships no macOS wheel on PyPI**
(across all its versions PyPI has only manylinux wheels + an sdist, and the sdist
pins NVIDIA-only deps that make it unsatisfiable on macOS). The only macOS `vllm`
wheel exists as a GitHub *release asset* — which `install.sh` pins by URL, but
which cannot go into a PyPI package's dependency metadata (PyPI forbids
direct-URL deps).

So today the supported install is still `install.sh` (which fetches vLLM core
and the plugin as prebuilt wheels — zero compile). Turning
`pip install vllm-metal` into a complete one-command install needs an **upstream**
change: vLLM must publish its macOS arm64 wheel to PyPI as part of its release
pipeline. Only then can vllm-metal flip `vllm` to a hard pinned dependency. This
is tracked in [#436](https://github.com/vllm-project/vllm-metal/issues/436).
