#!/usr/bin/env bash

set -euo pipefail

strip_args=()
[[ "${SLICER_STRIP:-1}" == 0 ]] || strip_args+=(--strip)

library_args=(--library-root slicer-src/build/Release)
for conan_root in slicer-src/deps/build/conan "${CONAN_HOME:-}"; do
  if [[ -n "$conan_root" && -d "$conan_root" ]]; then
    library_args+=(--library-root "$conan_root")
  fi
done

python3 tools/stage_bundle.py \
  --executable slicer-src/build/Release/CuraEngine \
  --executable slicer-src/build/CuraEngine \
  --name CuraEngine \
  --arch "${ARCH:?ARCH is required}" \
  --output slicer-src/build/slicer_out \
  "${library_args[@]}" \
  "${strip_args[@]}" \
  --json | tee slicer-src/build/slicer-bundle-report.json
