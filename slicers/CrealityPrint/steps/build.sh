#!/bin/bash
set -euo pipefail

bash ./tools/stamp_version_date.sh slicer-src
source_dir="$(cd -- slicer-src && pwd -P)"
deps_root="${DEPS_ENV_DIR:-}"
if [[ ! -d "$deps_root" ]]; then
  deps_root="$source_dir/deps/build/destdir"
fi
prefix="$deps_root/usr/local"

bash ./tools/build_cmake_target.sh \
  --source "$source_dir" \
  --target CrealityPrint \
  --generator Ninja \
  --gettext run_gettext.sh \
  -- \
  "-DCMAKE_PREFIX_PATH=$prefix" \
  -DSLIC3R_STATIC=1 \
  -DORCA_TOOLS=ON \
  -DGENERATE_ORCA_HEADER=0 \
  -DENABLE_BREAKPAD=ON \
  -DSLIC3R_GTK=3 \
  -DBBL_RELEASE_TO_PUBLIC=1 \
  -DBBL_INTERNAL_TESTING=0 \
  -DUPDATE_ONLINE_MACHINES=1
