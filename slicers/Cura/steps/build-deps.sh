#!/usr/bin/env bash

set -euo pipefail

# shellcheck source=conan-env.sh
source slicers/Cura/steps/conan-env.sh
cura_conan_env

jobs="${SLICER_JOBS:-$(nproc)}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLICER_JOBS must be a positive integer, got: $jobs" >&2
  exit 2
fi

"$CURA_CONAN_BIN" install slicer-src \
  --settings build_type=Release \
  --build=missing \
  --conf 'tools.build:skip_test=True' \
  --conf "tools.build:jobs=$jobs" \
  -o 'curaengine/*:enable_arcus=False' \
  -o 'curaengine/*:enable_benchmarks=False' \
  -o 'curaengine/*:enable_extensive_warnings=False' \
  -o 'curaengine/*:enable_plugins=False' \
  -o 'curaengine/*:enable_remote_plugins=False'
