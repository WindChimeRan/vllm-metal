# SPDX-License-Identifier: Apache-2.0
"""Launch-policy tests for the optional NAX prefill shader library."""

from __future__ import annotations

from pathlib import Path

import pytest

import vllm_metal.envs as envs
from vllm_metal.metal import _try_init_nax_library


class _FakeNaxOps:
    def __init__(
        self,
        *,
        supported: bool = True,
        ready: bool = True,
        load_error: Exception | None = None,
    ) -> None:
        self.supported = supported
        self.ready = ready
        self.load_error = load_error
        self.calls: list[tuple[str, str | None]] = []

    def nax_supported(self) -> bool:
        self.calls.append(("supported", None))
        return self.supported

    def init_nax_library(self, source: str) -> None:
        self.calls.append(("source", source))
        if self.load_error is not None:
            raise self.load_error

    def init_nax_library_path(self, path: str) -> None:
        self.calls.append(("path", path))
        if self.load_error is not None:
            raise self.load_error

    def nax_ready(self) -> bool:
        self.calls.append(("ready", None))
        return self.ready


def test_disable_nax_env_is_a_negative_override(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_METAL_DISABLE_NAX", raising=False)
    assert envs.VLLM_METAL_DISABLE_NAX is False

    monkeypatch.setenv("VLLM_METAL_DISABLE_NAX", "1")
    assert envs.VLLM_METAL_DISABLE_NAX is True
    assert "VLLM_METAL_NAX_PREFILL" not in envs.environment_variables


def test_disabled_override_skips_hardware_probe(tmp_path: Path) -> None:
    ops = _FakeNaxOps()
    lib = tmp_path / "paged_attention_nax_kern.metallib"
    lib.write_bytes(b"lib")

    assert not _try_init_nax_library(
        ops,  # type: ignore[arg-type]
        disabled=True,
        build_from_source=False,
        prebuilt_path=lib,
    )
    assert ops.calls == []


def test_supported_prebuilt_library_loads_automatically(tmp_path: Path) -> None:
    ops = _FakeNaxOps()
    lib = tmp_path / "paged_attention_nax_kern.metallib"
    lib.write_bytes(b"lib")

    assert _try_init_nax_library(
        ops,  # type: ignore[arg-type]
        disabled=False,
        build_from_source=False,
        prebuilt_path=lib,
    )
    assert ops.calls == [
        ("supported", None),
        ("path", str(lib)),
        ("ready", None),
    ]


@pytest.mark.parametrize("supported,has_library", [(False, True), (True, False)])
def test_unavailable_nax_keeps_fallback(
    tmp_path: Path, supported: bool, has_library: bool
) -> None:
    ops = _FakeNaxOps(supported=supported)
    lib = tmp_path / "paged_attention_nax_kern.metallib"
    if has_library:
        lib.write_bytes(b"lib")

    assert not _try_init_nax_library(
        ops,  # type: ignore[arg-type]
        disabled=False,
        build_from_source=False,
        prebuilt_path=lib,
    )
    assert not any(call[0] in {"path", "source"} for call in ops.calls)


@pytest.mark.parametrize("build_from_source", [False, True])
def test_optional_load_failure_warns_and_keeps_fallback(
    tmp_path: Path,
    monkeypatch,
    caplog,
    build_from_source: bool,
) -> None:
    ops = _FakeNaxOps(load_error=RuntimeError("bad optional library"))
    lib = tmp_path / "paged_attention_nax_kern.metallib"
    lib.write_bytes(b"lib")
    monkeypatch.setattr("vllm_metal.metal._build_nax_source", lambda: "nax source")

    assert not _try_init_nax_library(
        ops,  # type: ignore[arg-type]
        disabled=False,
        build_from_source=build_from_source,
        prebuilt_path=lib,
    )
    assert "using the non-NAX fallback" in caplog.text
