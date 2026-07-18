#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f slicer-src/build/Release/generators/conan_toolchain.cmake ]]; then
  ./slicers/Cura/steps/build-deps.sh
fi

deps="$PWD/slicer-src/deps/build"
export CONAN_HOME="$deps/conan"

"$deps/venv/bin/conan" build slicer-src \
  --settings build_type=Release \
  -o 'curaengine/*:enable_arcus=False' \
  -o 'curaengine/*:enable_plugins=False' \
  -o 'curaengine/*:enable_remote_plugins=False'
