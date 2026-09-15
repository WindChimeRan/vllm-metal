# SPDX-License-Identifier: Apache-2.0
"""Reuse a release wheel's native artifacts in a Python-editable checkout."""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from email import message_from_bytes
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import parse_tag, sys_tags

# Resolve the checkout even before an editable install exists.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_metal.metal import build  # noqa: E402


def install_prebuilt(wheel: zipfile.ZipFile) -> None:
    metadata_path = next(
        n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")
    )
    metadata = message_from_bytes(wheel.read(metadata_path))
    wheel_info = message_from_bytes(
        wheel.read(metadata_path.removesuffix("METADATA") + "WHEEL")
    )
    tags = {tag for value in wheel_info.get_all("Tag", []) for tag in parse_tag(value)}
    if not tags.intersection(sys_tags()):
        raise ValueError("The release wheel does not support this Python/platform.")

    requirements = {
        req.name: req
        for value in metadata.get_all("Requires-Dist", [])
        if (req := Requirement(value)).name in {"mlx", "nanobind"}
    }
    for name in ("mlx", "nanobind"):
        if (
            name not in requirements
            or str(requirements[name].specifier) != f"=={version(name)}"
        ):
            raise ValueError(
                f"The release wheel's {name} version does not match this environment."
            )

    native_dir = build.output_path().parent
    prefix = "vllm_metal/metal/"
    source_suffixes = {".py", ".cpp", ".h", ".metal"}
    local_sources = {
        prefix + path.relative_to(native_dir).as_posix(): path
        for path in native_dir.rglob("*")
        if path.suffix in source_suffixes
    }
    wheel_sources = {
        name
        for name in wheel.namelist()
        if name.startswith(prefix) and Path(name).suffix in source_suffixes
    }
    if local_sources.keys() != wheel_sources:
        raise ValueError(
            "The release wheel's native source files differ from this checkout."
        )
    for name, path in local_sources.items():
        if path.read_bytes() != wheel.read(name):
            raise ValueError(f"Native source differs from the release wheel: {name}")

    libraries = [*build.METALLIB_NAMES, build.NAX_METALLIB_NAME]
    paths = [build.output_path(), *(build.metallib_path(name) for name in libraries)]
    # Validate and read everything before replacing any existing artifact.
    artifacts = {path: wheel.read(prefix + path.name) for path in paths}
    stamps = {build.output_path(): build._input_hash(build._build_spec())}
    stamps.update(
        {
            build.metallib_path(name): build._metallib_digest(
                name, build._metallib_source(name)
            )
            for name in libraries
        }
    )
    for path, data in artifacts.items():
        path.write_bytes(data)
        build._stamp_path(path).write_text(stamps[path])
    print(
        "Installed matching prebuilt native artifacts; Python sources remain editable."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", help="Release wheel path or HTTPS URL")
    args = parser.parse_args()
    try:
        archive = args.wheel
        if archive.startswith("https://"):
            with urllib.request.urlopen(archive, timeout=60) as response:
                archive = io.BytesIO(response.read())
        with zipfile.ZipFile(archive) as wheel:
            install_prebuilt(wheel)
    except (OSError, ValueError, KeyError, StopIteration, zipfile.BadZipFile) as exc:
        raise SystemExit(
            f"Cannot reuse prebuilt kernels: {exc}\n"
            "Use ./install.sh --build to build native artifacts from this checkout."
        ) from exc


if __name__ == "__main__":
    main()
