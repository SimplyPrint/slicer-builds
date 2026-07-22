#!/usr/bin/env bash
set -euo pipefail

source_dir="slicer-src"
build_dir="$source_dir/deps/build"
generator="${CMAKE_GENERATOR:-Ninja}"
cmake_args=(-S "$source_dir/deps" -B "$build_dir")

if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
  cmake_args+=(-G "$generator")
fi

cmake "${cmake_args[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEP_WX_GTK3=ON

# SuperSlicer's dependency projects apply their own detected CPU count. A
# serial top-level build prevents nested parallel builds from oversubscribing.
case "${SLICER_GUI:-0}" in
  1 | true | TRUE | on | ON)
    cmake --build "$build_dir" --parallel 1 --target deps
    ;;
  *)
    # The old SuperSlicer dependency project has no package-exclusion option,
    # but exposes stable per-package targets. Avoid wxWidgets, OCCT, OpenCSG,
    # and the other GUI-only dependency closure for the CLI build.
    cmake --build "$build_dir" --parallel 1 --target \
      dep_Boost \
      dep_Cereal \
      dep_CGAL \
      dep_JPEG \
      dep_NLopt \
      dep_OpenVDB \
      dep_Qhull \
      dep_TBB
    ;;
esac
