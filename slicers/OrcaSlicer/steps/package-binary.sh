#!/usr/bin/env bash

set -euo pipefail

strip_args=()
[[ "${SLICER_STRIP:-1}" == 0 ]] || strip_args+=(--strip)

if [[ -z "${SLICER_RESOURCE_INCLUDES+x}" ]]; then
  export SLICER_RESOURCE_INCLUDES=$'info/**\nprofiles/BBL/cli_config.json\nshaders/**'
fi

python3 tools/stage_bundle.py \
  --executable slicer-src/build/src/Release/orca-slicer \
  --executable slicer-src/build/src/orca-slicer \
  --name orca-slicer \
  --arch "${ARCH:?ARCH is required}" \
  --output slicer-src/build/slicer_out \
  --resources slicer-src/resources \
  --library-root slicer-src/build/src \
  --library-root slicer-src/deps/build \
  "${strip_args[@]}" \
  --json | tee slicer-src/build/slicer-bundle-report.json
