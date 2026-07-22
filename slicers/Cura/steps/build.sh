#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f slicer-src/build/Release/generators/conan_toolchain.cmake ]]; then
  ./slicers/Cura/steps/build-deps.sh
fi

# shellcheck source=conan-env.sh
source slicers/Cura/steps/conan-env.sh
cura_conan_env

jobs="${SLICER_JOBS:-$(nproc)}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLICER_JOBS must be a positive integer, got: $jobs" >&2
  exit 2
fi

"$CURA_CONAN_BIN" build slicer-src \
  --settings build_type=Release \
  --conf 'tools.build:skip_test=True' \
  --conf "tools.build:jobs=$jobs" \
  -o 'curaengine/*:enable_arcus=False' \
  -o 'curaengine/*:enable_benchmarks=False' \
  -o 'curaengine/*:enable_extensive_warnings=False' \
  -o 'curaengine/*:enable_plugins=False' \
  -o 'curaengine/*:enable_remote_plugins=False'
