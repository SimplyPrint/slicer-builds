#!/usr/bin/env bash
set -euo pipefail
# https://github.com/prusa3d/PrusaSlicer/blob/master/doc/How%20to%20build%20-%20Linux%20et%20al.md

source_dir="slicer-src"
build_dir="$source_dir/deps/build"
generator="${CMAKE_GENERATOR:-Ninja}"
cmake_args=(-S "$source_dir/deps" -B "$build_dir")

# Keep an existing generator so restored CI caches remain reusable. New build
# directories use Ninja unless the caller selects another generator.
if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
  cmake_args+=(-G "$generator")
fi

case "${SLICER_GUI:-0}" in
  1 | true | TRUE | on | ON)
    package_excludes="Catch2|OCCT|OpenCSG"
    ;;
  *)
    # These packages are only used by the GUI, tests, sandboxes, or STEP import.
    package_excludes="CURL|Catch2|GLEW|NanoSVG|OCCT|OpenCSG|OpenSSL|wxWidgets"
    ;;
esac

cmake "${cmake_args[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEP_WX_GTK3=ON \
  -DPrusaSlicer_deps_PACKAGE_EXCLUDES="$package_excludes"

# Each ExternalProject already builds in parallel. Keeping this top-level target
# serial avoids multiplying the job count for concurrently-built dependencies.
cmake --build "$build_dir" --parallel 1 --target deps
