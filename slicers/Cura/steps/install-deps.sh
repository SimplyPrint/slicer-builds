#!/usr/bin/env bash

set -euo pipefail

sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  ninja-build \
  python3-venv

deps="$PWD/slicer-src/deps/build"
python3 -m venv "$deps/venv"
"$deps/venv/bin/pip" install --disable-pip-version-check "conan==2.15.1"

export CONAN_HOME="$deps/conan"
"$deps/venv/bin/conan" config install https://github.com/Ultimaker/conan-config.git
"$deps/venv/bin/conan" profile detect --force
