#!/bin/bash
# Common library functions for vllm-metal scripts

# Print an error message
error() {
  echo -e "Error: $*" >&2
}

# Print a success message
success() {
  echo -e "✓ $*"
}

# Print a section header
section() {
  echo "=== $* ==="
}

validate_vllm_release_tag() {
  local release_tag="$1"
  if [[ ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-].*)?$ ]]; then
    error "Invalid vLLM release tag: ${release_tag}"
    return 1
  fi
}

# Read the upstream vLLM release paired with this source revision.
read_vllm_release_tag() {
  local path="${1:-.github/vllm-release-tag.commit}"
  local release_tag

  if [ ! -r "$path" ]; then
    error "Missing vLLM release tag file: ${path}"
    return 1
  fi

  release_tag=$(tr -d '[:space:]' < "$path")
  validate_vllm_release_tag "$release_tag" || return 1
  printf '%s\n' "$release_tag"
}

# Check if running on Apple Silicon
is_apple_silicon() {
  [ "$(uname -m)" = "arm64" ]
}

# Require a native arm64 Python interpreter.
require_arm64_python() {
  local python_bin="${1:-python}"
  local machine

  if ! machine=$("$python_bin" -c "import platform; print(platform.machine())"); then
    error "Failed to inspect Python architecture using ${python_bin}."
    return 1
  fi

  if [ "$machine" != "arm64" ]; then
    error "vllm-metal requires native arm64 Python, got ${machine}. Remove the venv and rerun install.sh from an arm64 Python."
    return 1
  fi
}

# Ensure uv is installed
ensure_uv() {
  if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    if ! curl -LsSf "https://astral.sh/uv/0.9.18/install.sh" | sh; then
      error "Failed to install uv"
      return 1
    fi

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

# Ensure virtual environment exists and is activated
ensure_venv() {
  if [ ! -d "$1" ]; then
    section "Creating virtual environment"
    uv venv "$1" --clear --python 3.12 --seed
  fi

  # shellcheck source=/dev/null
  source "$1/bin/activate"
}

# Install dev dependencies
install_dev_deps() {
  section "Installing dependencies"
  uv pip install -e ".[dev]"
}

# Full development environment setup
setup_dev_env() {
  ensure_uv
  ensure_venv ".venv-vllm-metal"
  install_dev_deps
}

# Get version from pyproject.toml
get_version() {
  uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
}

# Check the compiler needed for prebuilt .metallib artifacts.
check_metal_toolchain() {
  section "Checking Metal toolchain"
  # Keep native artifacts aligned with the wheel's macOS 15 floor.
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"

  local tmpdir metal_src metal_lib
  tmpdir=$(mktemp -d)
  metal_src="${tmpdir}/probe.metal"
  metal_lib="${tmpdir}/probe.metallib"
  printf '[[kernel]] void _t() {}\n' > "${metal_src}"

  if ! xcrun -sdk macosx metal -o "${metal_lib}" "${metal_src}"; then
    error "Cannot compile Metal libraries. Resolve the compiler error above."
    echo "If the Metal component is missing, install it with: xcodebuild -downloadComponent MetalToolchain" >&2
    rm -rf "${tmpdir}"
    return 1
  fi

  success "Metal toolchain ready"
  rm -rf "${tmpdir}"
}

# Build the in-package native artifacts (the _paged_ops*.so and the required
# precompiled .metallib shader libraries, including NAX) into vllm_metal/metal/
# so `uv build` can bundle them as package data.
#
# `python` here is the venv interpreter activated by setup_dev_env, so mlx and
# nanobind are importable.
build_native_artifacts() {
  section "Building native Metal artifacts"
  # Official wheels require NAX, so reject an older SDK before compiling.
  if ! python -c \
    "from vllm_metal.metal.build import require_nax_sdk; require_nax_sdk()"; then
    return 1
  fi
  python -m vllm_metal.metal.build
}

# Fail unless the freshly built wheel actually bundles the prebuilt native
# artifacts: the _paged_ops*.so extension, three required metallibs, and NAX.
# setup.py's package data is what pulls these (gitignored)
# files in; if that ever regresses, the wheel would install fine but fail at
# first run with "Prebuilt native extension not found". The expected filenames
# are read from build.py so this guard never drifts from the runtime loader.
#
# Usage: verify_wheel_artifacts <path-to-wheel>
verify_wheel_artifacts() {
  local wheel="$1"
  section "Verifying wheel bundles native artifacts"

  local expected
  if ! expected=$(python -c "
from vllm_metal.metal.build import METALLIB_NAMES, NAX_METALLIB_NAME, metallib_path, output_path
print(output_path().name)
for _name in (*METALLIB_NAMES, NAX_METALLIB_NAME):
    print(metallib_path(_name).name)
"); then
    error "Failed to resolve expected native artifact names from build.py."
    return 1
  fi

  local contents name
  contents=$(unzip -Z1 "$wheel")
  while IFS= read -r name; do
    [ -z "$name" ] && continue
    if grep -qxF "vllm_metal/metal/$name" <<< "$contents"; then
      success "bundled: ${name}"
    else
      error "Wheel ${wheel} is missing native artifact: ${name}"
      error "Check the native package data in setup.py."
      return 1
    fi
  done <<< "$expected"

  local expected_minos paged_ops_name native_so actual_minos
  expected_minos=$(python -c "from vllm_metal.metal.build import MIN_MACOS_VERSION; print(MIN_MACOS_VERSION)")
  paged_ops_name=$(python -c "from vllm_metal.metal.build import output_path; print(output_path().name)")
  native_so=$(mktemp)
  if ! unzip -p "$wheel" "vllm_metal/metal/$paged_ops_name" > "$native_so"; then
    rm -f "$native_so"
    return 1
  fi
  actual_minos=$(otool -l "$native_so" | awk '$1 == "minos" { print $2; exit }') || actual_minos=""
  rm -f "$native_so"
  if [ "$actual_minos" != "$expected_minos" ]; then
    error "$paged_ops_name targets macOS ${actual_minos:-unknown}; expected ${expected_minos}."
    return 1
  fi
  success "$paged_ops_name: macOS $actual_minos"

  success "Wheel bundles all native artifacts"
}
