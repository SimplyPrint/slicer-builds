#!/usr/bin/env bash
set -euo pipefail
# https://github.com/prusa3d/PrusaSlicer/blob/master/doc/How%20to%20build%20-%20Linux%20et%20al.md

source_dir="slicer-src"
build_dir="$source_dir/build"
prefix="$source_dir/deps/build/destdir/usr/local"
generator="${CMAKE_GENERATOR:-Ninja}"
jobs="${SLICER_JOBS:-$(nproc)}"
cmake_args=(-S "$source_dir" -B "$build_dir")

if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
  cmake_args+=(-G "$generator")
fi

if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLICER_JOBS must be a positive integer, got: $jobs" >&2
  exit 2
fi

case "${SLICER_GUI:-0}" in
  1 | true | TRUE | on | ON) gui=ON ;;
  *) gui=OFF ;;
esac

cmake "${cmake_args[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$prefix" \
  -DSLIC3R_BUILD_SANDBOXES=OFF \
  -DSLIC3R_BUILD_TESTS=OFF \
  -DSLIC3R_DESKTOP_INTEGRATION=OFF \
  -DSLIC3R_ENABLE_FORMAT_STEP=OFF \
  -DSLIC3R_GTK=3 \
  -DSLIC3R_GUI="$gui" \
  -DSLIC3R_PCH="${SLICER_PCH:-ON}" \
  -DSLIC3R_STATIC=ON

cmake --build "$build_dir" --parallel "$jobs" --target PrusaSlicer
