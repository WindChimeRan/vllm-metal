#!/bin/bash

# Build entrypoint for the PyPI channel. Separate from release.sh because that
# one dev-stamps the version and cuts a GitHub release; here the wheel in dist/
# keeps pyproject.toml's version and the workflow uploads it.
main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env
  build_wheel
}

main "$@"
