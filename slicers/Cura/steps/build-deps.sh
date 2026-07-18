#!/usr/bin/env bash

set -euo pipefail

deps="$PWD/slicer-src/deps/build"
export CONAN_HOME="$deps/conan"

"$deps/venv/bin/conan" install slicer-src \
  --settings build_type=Release \
  --build=missing \
  -o 'curaengine/*:enable_arcus=False' \
  -o 'curaengine/*:enable_plugins=False' \
  -o 'curaengine/*:enable_remote_plugins=False'
