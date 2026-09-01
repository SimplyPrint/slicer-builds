#!/usr/bin/env bash

set -euo pipefail

strip_args=()
[[ "${SLICER_STRIP:-1}" == 0 ]] || strip_args+=(--strip)
output_dir="${SLICER_PACKAGE_DIR:-slicer-src/build/slicer_out}"
resource_args=()
if [[ -d slicer-src/src/slic3r-app-launcher ]]; then
  resource_args+=(--resources slicer-src/resources)
fi

python3 tools/stage_bundle.py \
  --executable slicer-src/build/src/prusa-slicer \
  --executable slicer-src/build/src/Release/prusa-slicer \
  --executable slicer-src/build/src/slic3r-app-launcher/slic3r-app-launcher \
  --executable slicer-src/build/src/slic3r-app-launcher/Release/slic3r-app-launcher \
  --name prusa-slicer \
  --arch "${ARCH:?ARCH is required}" \
  --output "$output_dir" \
  "${resource_args[@]}" \
  --library-root slicer-src/build/src \
  --library-root slicer-src/deps/build \
  "${strip_args[@]}" \
  --json | tee slicer-src/build/slicer-bundle-report.json
