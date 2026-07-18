#!/usr/bin/env bash

set -euo pipefail

CURA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURA_REPO_ROOT="$(cd "${CURA_SCRIPT_DIR}/../../.." && pwd)"

# shellcheck source=../version.env
source "${CURA_REPO_ROOT}/slicers/Cura/version.env"

CURA_ENGINE_SOURCE_DIR="${CURA_ENGINE_SOURCE_DIR:-${CURA_REPO_ROOT}/slicer-src}"
CURA_RESOURCES_SOURCE_DIR="${CURA_RESOURCES_SOURCE_DIR:-${CURA_REPO_ROOT}/cura-resources}"
CURA_CONAN_CONFIG_DIR="${CURA_CONAN_CONFIG_DIR:-${CURA_REPO_ROOT}/.cura-conan-config}"
CURA_VENV_DIR="${CURA_VENV_DIR:-${CURA_ENGINE_SOURCE_DIR}/.venv}"
CURA_CONAN_HOME="${CURA_CONAN_HOME:-${CURA_REPO_ROOT}/.cura-conan-home}"
CURA_CONAN_GRAPH_FILE="${CURA_CONAN_GRAPH_FILE:-${CURA_ENGINE_SOURCE_DIR}/build/conan-install-graph.json}"

export CONAN_HOME="${CURA_CONAN_HOME}"

require_ref() {
  local directory="$1"
  local expected_ref="$2"
  local description="$3"
  local actual_ref

  actual_ref="$(git -C "${directory}" rev-parse HEAD)"
  if [[ "${actual_ref}" != "${expected_ref}" ]]; then
    echo "${description} is at ${actual_ref}; expected pinned ref ${expected_ref}." >&2
    return 1
  fi
}

clone_pinned_source() {
  local repository="$1"
  local ref="$2"
  local target="$3"

  if [[ ! -d "${target}/.git" ]]; then
    git clone --filter=blob:none --no-checkout "https://github.com/${repository}.git" "${target}"
  fi

  if ! git -C "${target}" cat-file -e "${ref}^{commit}" 2>/dev/null; then
    git -C "${target}" fetch --depth 1 origin "${ref}"
  fi
  git -C "${target}" checkout --detach "${ref}"
  require_ref "${target}" "${ref}" "${repository}"
}

prepare_pinned_sources() {
  require_ref "${CURA_ENGINE_SOURCE_DIR}" "${CURA_ENGINE_REF}" "${CURA_ENGINE_REPO}"
  clone_pinned_source "${CURA_RESOURCES_REPO}" "${CURA_RESOURCES_REF}" "${CURA_RESOURCES_SOURCE_DIR}"
}

activate_conan() {
  if [[ ! -x "${CURA_VENV_DIR}/bin/conan" ]]; then
    echo "Conan environment is missing. Run slicers/Cura/steps/install-deps.sh first." >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "${CURA_VENV_DIR}/bin/activate"
}

conan_options() {
  # The cloud slicer communicates exclusively through CuraEngine's command-line
  # protocol. Arcus and the plug-in hosts are not needed and would add shared
  # runtime services to the otherwise self-contained worker bundle.
  printf '%s\n' \
    '-o' 'curaengine/*:enable_arcus=False' \
    '-o' 'curaengine/*:enable_plugins=False' \
    '-o' 'curaengine/*:enable_remote_plugins=False'
}
