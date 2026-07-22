#!/bin/bash
set -euo pipefail

bash ./tools/stamp_version_date.sh slicer-src
source_dir="$(cd -- slicer-src && pwd -P)"
pch=ON
if [[ "${SLICER_PCH:-ON}" == "OFF" ]]; then
  pch=OFF
fi

export CMAKE_POLICY_VERSION_MINIMUM=3.5
read -r -a extra_args <<< "${ORCA_EXTRA_BUILD_ARGS:-}"
if [[ -n "${ORCA_UPDATER_SIG_KEY:-}" ]]; then
  extra_args+=("-DORCA_UPDATER_SIG_KEY=${ORCA_UPDATER_SIG_KEY}")
fi

bash ./tools/build_cmake_target.sh \
  --source "$source_dir" \
  --target OrcaSlicer \
  --generator "Ninja Multi-Config" \
  --config Release \
  --gettext scripts/run_gettext.sh \
  --auto-compiler-cache \
  -- \
  "-DSLIC3R_PCH=$pch" \
  "${extra_args[@]}" \
  -DORCA_TOOLS=OFF
