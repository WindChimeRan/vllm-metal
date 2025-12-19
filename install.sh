#!/bin/bash

fetch_latest_release() {
  local repo_owner="$1"
  local repo_name="$2"

  echo "Fetching latest release..." >&2

  local latest_release_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases/latest"
  local release_data

  if ! release_data=$(curl -fsSL "$latest_release_url" 2>&1); then
    error "Failed to fetch release information."
    echo "Please check your internet connection and try again." >&2
    exit 1
  fi

  if [[ -z "$release_data" ]] || [[ "$release_data" == *"Not Found"* ]]; then
    error "No releases found for this repository."
    echo "Please visit https://github.com/${repo_owner}/${repo_name}/releases" >&2
    exit 1
  fi

  echo "$release_data"
}

extract_wheel_url() {
  local release_data="$1"

  python3 -c "
import sys
import json
try:
    data = json.loads('''$release_data''')
    assets = data.get('assets', [])
    for asset in assets:
        name = asset.get('name', '')
        if name.endswith('.whl'):
            print(asset.get('browser_download_url', ''))
            break
except Exception as e:
    print('', file=sys.stderr)
"
}

download_and_install_wheel() {
  local wheel_url="$1"
  local package_name="$2"

  local wheel_name
  wheel_name=$(basename "$wheel_url")
  echo "Latest release: $wheel_name"
  success "Found latest release"

  local tmp_dir
  tmp_dir=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp_dir'" EXIT

  echo ""
  echo "Downloading wheel..."
  local wheel_path="$tmp_dir/$wheel_name"

  if ! curl -fsSL "$wheel_url" -o "$wheel_path"; then
    error "Failed to download wheel."
    exit 1
  fi

  success "Downloaded wheel"

  # Install vllm-metal base package first
  if ! uv pip install --upgrade "$wheel_path"; then
    error "Failed to install ${package_name}."
    exit 1
  fi

  # Install vllm separately with --no-deps to avoid pulling in CUDA dependencies
  # that don't work on macOS/Apple Silicon
  echo ""
  echo "Installing vllm (without CUDA dependencies)..."
  if ! uv pip install --upgrade --no-deps vllm; then
    error "Failed to install vllm."
    exit 1
  fi

  # Install vllm's macOS-compatible dependencies
  if ! uv pip install --upgrade \
    msgspec \
    cloudpickle \
    prometheus-client \
    fastapi \
    uvicorn \
    uvloop \
    pydantic \
    pillow \
    prometheus_fastapi_instrumentator \
    tiktoken \
    lm-format-enforcer \
    outlines \
    typing_extensions \
    filelock \
    py-cpuinfo \
    aiohttp \
    openai \
    einops \
    importlib_metadata \
    mistral_common \
    pyyaml \
    requests \
    tqdm \
    sentencepiece \
    compressed-tensors \
    gguf \
    partial-json-parser \
    blake3 \
    cbor2 \
    pyzmq \
    cachetools \
    regex \
    protobuf \
    python-multipart \
    "fastapi[standard]" \
    "llguidance>=1.3.0,<1.4.0" \
    "outlines_core==0.2.11" \
    "lark==1.2.2" \
    "xgrammar>=0.1.27" \
    "opencv-python-headless>=4.11.0" \
    six \
    "setuptools>=77.0.3,<81.0.0" \
    "depyf>=0.20.0" \
    watchfiles \
    python-json-logger \
    scipy \
    ninja \
    pybase64 \
    setproctitle \
    mlx \
    mlx-lm \
    "openai-harmony>=0.0.3" \
    "anthropic>=0.71.0" \
    "model-hosting-container-standards>=0.1.9,<1.0.0" \
    "tokenizers>=0.21.1" \
    "numba>=0.61.0"; then
    error "Failed to install vllm dependencies."
    exit 1
  fi
}

main() {
  set -eu -o pipefail

  local repo_owner="vllm-project"
  local repo_name="vllm-metal"
  local package_name="vllm-metal"

  # Source shared library functions
  # Try local lib.sh first (when running ./install.sh), fall back to remote (when piped from curl)
  local local_lib=""
  if [[ -n "${BASH_SOURCE[0]}" ]]; then
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local_lib="$script_dir/scripts/lib.sh"
  fi

  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    # shellcheck source=/dev/null
    source "$local_lib"
  else
    # Fetch from remote (curl | bash case)
    local lib_url="https://raw.githubusercontent.com/$repo_owner/$repo_name/main/scripts/lib.sh"
    local lib_tmp
    lib_tmp=$(mktemp)
    if ! curl -fsSL "$lib_url" -o "$lib_tmp"; then
      echo "Error: Failed to fetch lib.sh from $lib_url" >&2
      rm -f "$lib_tmp"
      exit 1
    fi
    # shellcheck source=/dev/null
    source "$lib_tmp"
    rm -f "$lib_tmp"
  fi

  is_apple_silicon
  if ! ensure_uv; then
    exit 1
  fi

  local release_data
  release_data=$(fetch_latest_release "$repo_owner" "$repo_name")

  local wheel_url
  wheel_url=$(extract_wheel_url "$release_data")

  if [[ -z "$wheel_url" ]]; then
    error "No wheel file found in the latest release."
    exit 1
  fi

  download_and_install_wheel "$wheel_url" "$package_name"

  echo ""
  success "Installation complete!"
  echo ""
  echo "To use vllm, activate the virtual environment:"
  echo "  source .venv/bin/activate"
  echo ""
  echo "Or add the venv to your PATH:"
  echo "  export PATH=\"\$PWD/.venv/bin:\$PATH\""
}

main "$@"

