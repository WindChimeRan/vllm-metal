# SPDX-License-Identifier: Apache-2.0
"""Reuse a release wheel's native artifacts in a Python-editable checkout."""

import io
import sys
import urllib.request
import zipfile
from email import message_from_bytes
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import parse_tag, sys_tags

# Always inspect the checkout being installed.
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

    pins = {
        req.name: str(req.specifier)
        for value in metadata.get_all("Requires-Dist", [])
        if (req := Requirement(value)).name in {"mlx", "nanobind"}
    }
    for name in ("mlx", "nanobind"):
        if pins.get(name) != f"=={version(name)}":
            raise ValueError(f"The release wheel requires a different {name} version.")

    native_dir = build.output_path().parent
    prefix = "vllm_metal/metal/"
    source_suffixes = {".py", ".cpp", ".h", ".metal"}
    sources = {
        prefix + path.relative_to(native_dir).as_posix(): path.read_bytes()
        for path in native_dir.rglob("*")
        if path.suffix in source_suffixes
    }
    wheel_sources = {
        name: wheel.read(name)
        for name in wheel.namelist()
        if name.startswith(prefix) and Path(name).suffix in source_suffixes
    }
    if sources != wheel_sources:
        raise ValueError(
            "The release wheel's native sources differ from this checkout."
        )

    stamps = {build.output_path(): build._input_hash(build._build_spec())}
    for name in (*build.METALLIB_NAMES, build.NAX_METALLIB_NAME):
        stamps[build.metallib_path(name)] = build._metallib_digest(
            name, build._metallib_source(name)
        )
    # Validate and read everything before replacing any existing artifact.
    artifacts = {path: wheel.read(prefix + path.name) for path in stamps}
    for path, data in artifacts.items():
        path.write_bytes(data)
        build._stamp_path(path).write_text(stamps[path])
    print(
        "Installed matching prebuilt native artifacts; Python sources remain editable."
    )


if __name__ == "__main__":
    # install.sh supplies the selected release's wheel URL.
    try:
        with urllib.request.urlopen(sys.argv[1], timeout=60) as response:
            archive = io.BytesIO(response.read())
        with zipfile.ZipFile(archive) as wheel:
            install_prebuilt(wheel)
    except (OSError, ValueError, KeyError, StopIteration, zipfile.BadZipFile) as exc:
        raise SystemExit(
            f"Cannot reuse prebuilt kernels: {exc}\n"
            "Use ./install.sh --build to build native artifacts from this checkout."
        ) from exc
