#!/usr/bin/env bash

set -euo pipefail

sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  ninja-build \
  python3-venv

deps="$PWD/slicer-src/deps/build"
if [[ -z "${CONAN:-}" && ! -x "$deps/venv/bin/conan" ]]; then
  python3 -m venv "$deps/venv"
  "$deps/venv/bin/pip" install --disable-pip-version-check "conan==2.15.1"
fi

# shellcheck source=conan-env.sh
source slicers/Cura/steps/conan-env.sh
cura_conan_env
