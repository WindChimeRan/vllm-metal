#!/bin/bash
#
# Build and publish dev or stable wheels through the same verified path.
# Missing tags are created at the exact commit that produced the wheel.

usage() {
  cat <<'EOF'
Usage: scripts/release.sh --channel <dev|stable> [--version <X.Y.Z>]

Options:
  --channel dev      Timestamped prerelease build (default; used by CI on main).
  --channel stable   Release build. Version comes from pyproject.toml unless
                     --version is given. An "rc" in the version publishes a
                     prerelease.
  --version X.Y.Z    Override the version. Only valid with --channel stable,
                     e.g. --version 0.4.0rc1.
  -h, --help         Show this help.
EOF
}

validate_stable_version() {
  local v="$1"
  if [[ ! "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+|\.post[0-9]+)?$ ]]; then
    error "Stable version must be X.Y.Z, X.Y.ZrcN, or X.Y.Z.postN; got '${v}'."
    return 1
  fi
}

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  local channel="dev" version_override=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --channel)
        if [ $# -lt 2 ]; then
          error "--channel requires a value."
          exit 1
        fi
        channel="${2:-}"
        shift 2
        ;;
      --version)
        if [ $# -lt 2 ]; then
          error "--version requires a value."
          exit 1
        fi
        version_override="${2:-}"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        error "Unknown argument: $1"
        usage >&2
        exit 1
        ;;
    esac
  done

  case "$channel" in
    dev|stable) ;;
    *)
      error "--channel must be 'dev' or 'stable', got '${channel}'."
      exit 1
      ;;
  esac

  if [ -n "$version_override" ] && [ "$channel" != "stable" ]; then
    error "--version is only valid with --channel stable."
    exit 1
  fi

  if [ -n "$version_override" ]; then
    validate_stable_version "$version_override"
  fi

  local vllm_release_tag
  vllm_release_tag=$(read_vllm_release_tag)
  echo "vLLM release: $vllm_release_tag"

  setup_dev_env

  # Package all native artifacts before building the wheel.
  ensure_metal_toolchain
  build_native_artifacts

  local version prerelease=0
  if [ "$channel" = "dev" ]; then
    local base_version
    base_version=$(get_version)
    version="${base_version}.dev$(date -u +%Y%m%d%H%M%S)"
    prerelease=1
  else
    version="${version_override:-$(get_version)}"
    validate_stable_version "$version"
    if [[ "$version" == *rc* ]]; then
      prerelease=1
    fi
  fi
  echo "Channel: $channel"
  echo "Building version: $version"

  # Stamp the ephemeral checkout; maturin reads [project].version.
  sed -i '' -E "s/^version = .*/version = \"${version}\"/" pyproject.toml

  section "Building wheel"
  uv build

  # Abort before publishing if maturin omitted a native artifact.
  local wheels=(dist/*.whl)
  if [ ! -f "${wheels[0]}" ]; then
    error "No wheel found in dist/ after uv build."
    exit 1
  fi
  verify_wheel_artifacts "${wheels[0]}"

  local tag
  tag="v${version}"
  echo "Generated tag: $tag"

  section "Creating GitHub release"
  local release_args=(
    --title "$tag"
    --generate-notes
    --target "$(git rev-parse HEAD)"
  )
  if [ "$channel" = "stable" ]; then
    release_args+=(--draft)
  fi
  if [ "$prerelease" -eq 1 ]; then
    release_args+=(--prerelease)
  fi
  gh release create "$tag" "${release_args[@]}" dist/*.whl
}

main "$@"
