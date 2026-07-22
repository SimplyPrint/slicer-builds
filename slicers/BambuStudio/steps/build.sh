#!/bin/bash
set -euo pipefail
# https://github.com/bambulab/BambuStudio/wiki/Linux-Compile-Guide

bash ./tools/stamp_version_date.sh slicer-src
pushd slicer-src

build_args=(-sr)
if [[ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
  build_args+=(-f)
fi
./BuildLinux.sh "${build_args[@]}"

popd
