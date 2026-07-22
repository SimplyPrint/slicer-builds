#!/bin/bash
set -euo pipefail

bash ./tools/stamp_version_date.sh slicer-src
source_dir="$(cd -- slicer-src && pwd -P)"
prefix="$source_dir/deps/build/destdir/usr/local"
pch=ON
if [[ "${SLICER_PCH:-ON}" == "OFF" ]]; then
  pch=OFF
fi

export CMAKE_POLICY_VERSION_MINIMUM=3.5
read -r -a extra_args <<< "${ORCA_EXTRA_BUILD_ARGS:-}"
extra_args+=(
  -DSLIC3R_GTK=3
  -DBBL_RELEASE_TO_PUBLIC=1
  -DBBL_INTERNAL_TESTING=0
)
if [[ -n "${ORCA_UPDATER_SIG_KEY:-}" ]]; then
  extra_args+=("-DORCA_UPDATER_SIG_KEY=${ORCA_UPDATER_SIG_KEY}")
fi

# The upstream wrapper names AnycubicSlicerNext here, but the generated Ninja
# graph exposes the executable output target as orca-slicer.
bash ./tools/build_cmake_target.sh \
  --source "$source_dir" \
  --target orca-slicer \
  --generator "Ninja Multi-Config" \
  --config Release \
  --gettext scripts/run_gettext.sh \
  -- \
  "-DSLIC3R_PCH=$pch" \
  "-DCMAKE_PREFIX_PATH=$prefix" \
  -DSLIC3R_STATIC=1 \
  "${extra_args[@]}" \
  -DORCA_TOOLS=OFF
