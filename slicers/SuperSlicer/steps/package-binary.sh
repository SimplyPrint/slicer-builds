#!/usr/bin/env bash

set -euo pipefail

strip_args=()
[[ "${SLICER_STRIP:-1}" == 0 ]] || strip_args+=(--strip)

library_args=()
for library_root in \
  slicer-src/build/bin \
  slicer-src/build/src \
  slicer-src/deps/build; do
  if [[ -d "$library_root" ]]; then
    library_args+=(--library-root "$library_root")
  fi
done

python3 tools/stage_bundle.py \
  --executable slicer-src/build/bin/superslicer \
  --executable slicer-src/build/src/superslicer \
  --executable slicer-src/build/package/bin/superslicer \
  --executable slicer-src/build/superslicer \
  --name superslicer \
  --arch "${ARCH:?ARCH is required}" \
  --output slicer-src/build/slicer_out \
  "${library_args[@]}" \
  "${strip_args[@]}" \
  --json | tee slicer-src/build/slicer-bundle-report.json
