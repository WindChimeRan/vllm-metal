# SPDX-License-Identifier: Apache-2.0
"""Prebuilt reuse must preserve ABI compatibility and kernel staleness checks."""

import io
import shutil
import zipfile
from importlib.metadata import version

import pytest
from packaging.tags import sys_tags

from scripts.install_prebuilt import install_prebuilt
from vllm_metal.metal import build


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    native = tmp_path / "metal"
    shutil.copytree(
        build.output_path().parent,
        native,
        ignore=shutil.ignore_patterns("*.so", "*.metallib", "*.sha256", "__pycache__"),
    )
    for attr in ("_SRC", "_BUILD", "_CONSTANTS", "_OUT", "_HASH"):
        monkeypatch.setattr(build, attr, native / getattr(build, attr).name)
    monkeypatch.setattr(build, "_THIS_DIR", native)
    return native


def release_wheel(native, *, mlx_version=None, tag=None, omit=None):
    contents = io.BytesIO()
    with zipfile.ZipFile(contents, "w") as wheel:
        wheel.writestr(
            "vllm_metal-1.0.dist-info/METADATA",
            f"Requires-Dist: mlx=={mlx_version or version('mlx')}\n"
            f"Requires-Dist: nanobind=={version('nanobind')}\n",
        )
        wheel.writestr(
            "vllm_metal-1.0.dist-info/WHEEL", f"Tag: {tag or next(sys_tags())}\n"
        )
        for path in native.rglob("*"):
            if path.suffix in {".py", ".cpp", ".h", ".metal"}:
                wheel.write(
                    path, "vllm_metal/metal/" + path.relative_to(native).as_posix()
                )
        names = [build.output_path().name] + [
            build.metallib_path(n).name
            for n in (*build.METALLIB_NAMES, build.NAX_METALLIB_NAME)
        ]
        for name in names:
            if name != omit:
                wheel.writestr("vllm_metal/metal/" + name, b"prebuilt")
    return zipfile.ZipFile(contents)


def test_reused_artifacts_detect_later_kernel_edits(checkout):
    with release_wheel(checkout) as wheel:
        install_prebuilt(wheel)
    assert build.output_path().read_bytes() == b"prebuilt"
    assert build.stale_artifacts() == []

    build._SRC.write_text(build._SRC.read_text() + "\n// kernel changed\n")
    assert build.stale_artifacts() == [build.output_path()]


@pytest.mark.parametrize("mismatch", ["mlx", "python", "source", "missing"])
def test_incompatible_wheel_preserves_existing_artifacts(checkout, mismatch):
    build.output_path().write_bytes(b"existing")
    options = {
        "mlx": {"mlx_version": "0.0.0"},
        "python": {"tag": "cp310-cp310-macosx_15_0_arm64"},
        "source": {},
        "missing": {"omit": build.metallib_path(build.METALLIB_NAMES[-1]).name},
    }
    with release_wheel(checkout, **options[mismatch]) as wheel:
        if mismatch == "source":
            build._SRC.write_text("changed native source")
        with pytest.raises((ValueError, KeyError)):
            install_prebuilt(wheel)
    assert build.output_path().read_bytes() == b"existing"
    assert not list(checkout.glob("*.sha256"))
