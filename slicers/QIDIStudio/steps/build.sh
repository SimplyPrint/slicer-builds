#!/bin/bash
set -euo pipefail
# https://github.com/bambulab/BambuStudio/wiki/Linux-Compile-Guide

bash ./tools/stamp_version_date.sh slicer-src
source_dir="$(cd -- slicer-src && pwd -P)"
prefix="$source_dir/deps/build/destdir/usr/local"
if [[ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
  free_mem_gb="$(free -g -t | grep 'Mem' | rev | cut -d" " -f1 | rev)"
  max_threads=$((free_mem_gb * 10 / 25))
  export CMAKE_BUILD_PARALLEL_LEVEL="$((max_threads < 1 ? 1 : max_threads))"
fi

bash ./tools/build_cmake_target.sh \
  --source "$source_dir" \
  --target QIDIStudio \
  --generator Ninja \
  -- \
  "-DCMAKE_PREFIX_PATH=$prefix" \
  -DSLIC3R_STATIC=1 \
  -DSLIC3R_GTK=3 \
  -DQDT_RELEASE_TO_PUBLIC=0 \
  -DQDT_INTERNAL_TESTING=0
