#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ "${CURA_SKIP_SYSTEM_DEPS:-false}" != "true" ]]; then
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    patchelf \
    python3 \
    python3-venv
fi

if [[ ! -x "${CURA_VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${CURA_VENV_DIR}"
fi

"${CURA_VENV_DIR}/bin/python" -m pip install --disable-pip-version-check \
  "conan==${CURA_CONAN_VERSION}"

clone_pinned_source "${CURA_CONAN_CONFIG_REPO}" "${CURA_CONAN_CONFIG_REF}" "${CURA_CONAN_CONFIG_DIR}"

activate_conan
conan config install "${CURA_CONAN_CONFIG_DIR}"
conan profile detect --force
conan profile show
