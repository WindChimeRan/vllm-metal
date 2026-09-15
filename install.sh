#!/bin/bash

# Stable uses /releases/latest; dev selects the newest .dev tag.
fetch_release() {
  local repo_owner="$1"
  local repo_name="$2"
  local channel="$3"

  echo "Fetching ${channel} release..." >&2

  local api_url release_data
  if [[ "$channel" == "stable" ]]; then
    api_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases/latest"
  else
    api_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases?per_page=30"
  fi

  if ! release_data=$(curl -fsSL "$api_url"); then
    if [[ "$channel" == "stable" ]]; then
      error "Failed to fetch the latest stable release."
      echo "There may not be one yet. Retry with --dev for the latest development build." >&2
    else
      error "Failed to fetch release information."
      echo "Please check your internet connection and try again." >&2
    fi
    exit 1
  fi

  if [[ -z "$release_data" ]]; then
    error "No releases found for this repository."
    echo "Please visit https://github.com/${repo_owner}/${repo_name}/releases" >&2
    exit 1
  fi

  echo "$release_data"
}

# Print "<tag>\n<wheel url>" from a GitHub release payload on stdin.
extract_wheel_url() {
  local channel="$1"

  CHANNEL="$channel" python3 -c '
import json
import os
import sys

channel = os.environ["CHANNEL"]
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

# /releases/latest returns one release; /releases returns a list, newest first.
if isinstance(data, dict):
    releases = [data]
elif channel == "stable":
    releases = data
else:
    releases = [r for r in data if ".dev" in (r.get("tag_name") or "")]

for release in releases:
    for asset in release.get("assets", []):
        if (asset.get("name") or "").endswith(".whl"):
            print(release.get("tag_name", ""))
            print(asset.get("browser_download_url", ""))
            sys.exit(0)
'
}

fetch_release_vllm_tag() {
  local repo_owner="$1"
  local repo_name="$2"
  local release_tag="$3"
  local metadata_url metadata

  metadata_url="https://raw.githubusercontent.com/${repo_owner}/${repo_name}/${release_tag}/.github/vllm-release-tag.commit"
  if ! metadata=$(curl -fsSL "$metadata_url"); then
    error "Could not read the compatible vLLM release for ${release_tag}."
    return 1
  fi
  metadata=$(printf '%s' "$metadata" | tr -d '[:space:]')

  if ! validate_vllm_release_tag "$metadata"; then
    error "Release ${release_tag} has invalid vLLM metadata."
    return 1
  fi
  printf '%s\n' "$metadata"
}

install_vllm() {
  local vllm_release_tag="$1"
  local vllm_version="${vllm_release_tag#v}"
  local vllm_wheel_url="https://github.com/vllm-project/vllm/releases/download/${vllm_release_tag}/vllm-${vllm_version}%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl"

  echo ""
  section "Installing vLLM core"
  echo "Wheel: vLLM ${vllm_version} (prebuilt macOS arm64)"

  if ! uv pip install "$vllm_wheel_url"; then
    error "Failed to install vLLM core from ${vllm_wheel_url}"
    echo "Please check your internet connection and try again." >&2
    exit 1
  fi

  success "Installed vLLM core"
}

main() {
  set -eu -o pipefail

  local repo_owner="vllm-project"
  local repo_name="vllm-metal"
  local package_name="vllm-metal"

  # Override the default dev channel with --stable or VLLM_METAL_CHANNEL.
  local channel="${VLLM_METAL_CHANNEL:-dev}"
  local mode="wheel"

  for arg in "$@"; do
    case "$arg" in
      --dev)
        channel="dev"
        ;;
      --stable)
        channel="stable"
        ;;
      --editable|--build)
        if [[ "$mode" != "wheel" ]]; then
          echo "Choose either --editable or --build." >&2
          exit 1
        fi
        mode="${arg#--}"
        ;;
      -h|--help)
        cat <<'EOF'
Usage: install.sh [--dev | --stable] [--editable | --build]

Options:
      --dev         Install the latest development build cut from main.
                    This is the default and the currently recommended channel.
      --stable      Install the latest tagged stable release. Stable releases
                    are cut by hand and may lag behind the dev channel.
      --editable    Install this checkout's Python sources with prebuilt kernels.
      --build       Install this checkout and compile its native kernels.
  -h, --help        Show this help.

The channel can also be set with VLLM_METAL_CHANNEL=dev|stable.
The default installs release wheels into ~/.venv-vllm-metal without a compiler.
Contributor modes use .venv-vllm-metal in the checkout; only --build needs compilers.
EOF
        exit 0
        ;;
      *)
        echo "Unknown argument: $arg" >&2
        echo "Run with --help for usage." >&2
        exit 1
        ;;
    esac
  done

  case "$channel" in
    dev|stable) ;;
    *)
      echo "Invalid channel: $channel (expected 'dev' or 'stable')." >&2
      exit 1
      ;;
  esac

  # Load shared helpers from beside this script, or fetch them for curl | bash.
  local local_lib=""
  local script_dir=""
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" && pwd)"
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

  if ! is_apple_silicon; then
    error "vllm-metal requires Apple Silicon arm64. Detected: $(uname -m)."
    exit 1
  fi

  if ! ensure_uv; then
    exit 1
  fi

  local venv="$HOME/.venv-vllm-metal"
  if [[ "$mode" != "wheel" ]]; then
    if [[ -z "$local_lib" || ! -f "$local_lib" ]]; then
      error "--${mode} must be run from a source checkout."
      exit 1
    fi
    cd "$script_dir"
    venv="$script_dir/.venv-vllm-metal"
  fi
  ensure_venv "$venv"
  if ! require_arm64_python python; then
    exit 1
  fi

  local release_data selected release_tag wheel_url vllm_release_tag
  if [[ "$mode" != "build" ]]; then
    release_data=$(fetch_release "$repo_owner" "$repo_name" "$channel")

    # extract_wheel_url prints the tag on the first line, the URL on the second.
    selected=$(printf '%s' "$release_data" | extract_wheel_url "$channel")
    release_tag=$(printf '%s' "$selected" | sed -n '1p')
    wheel_url=$(printf '%s' "$selected" | sed -n '2p')

    if [[ "$channel" == "stable" &&
          ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(\.post[0-9]+)?$ ]]; then
      error "No stable release is available yet. Use --dev for the latest development build."
      exit 1
    fi

    if [[ -z "$wheel_url" ]]; then
      error "No wheel file found in the latest ${channel} release."
      exit 1
    fi
  fi

  if [[ "$mode" == "wheel" ]]; then
    vllm_release_tag=$(fetch_release_vllm_tag "$repo_owner" "$repo_name" "$release_tag")
    install_vllm "$vllm_release_tag"
    echo "Release: ${release_tag}"
    if ! uv pip install "$wheel_url"; then
      error "Failed to install ${package_name}."
      exit 1
    fi
  else
    if [[ "$mode" == "build" ]]; then
      check_metal_toolchain
    fi
    vllm_release_tag=$(read_vllm_release_tag)
    install_vllm "$vllm_release_tag"
    install_dev_deps
    if [[ "$mode" == "editable" ]]; then
      python scripts/install_prebuilt.py "$wheel_url"
    else
      build_native_artifacts
    fi
  fi

  echo ""
  success "Installation complete!"
  echo ""
  echo "To use vllm, activate the virtual environment:"
  echo "  source $venv/bin/activate"
  echo ""
  echo "Or add the venv to your PATH:"
  echo "  export PATH=\"$venv/bin:\$PATH\""
}

main "$@"
