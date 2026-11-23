#!/usr/bin/env bash

set -euo pipefail

strip_args=()
[[ "${SLICER_STRIP:-1}" == 0 ]] || strip_args+=(--strip)

if [[ -z "${SLICER_RESOURCE_INCLUDES+x}" ]]; then
  export SLICER_RESOURCE_INCLUDES=$'info/**\nprofiles/BBL/cli_config.json\nshaders/**'
fi

# OrcaSlicer 2.5 links the libpython its deps build compiles (for plugins).
# The build runner has a system libpython3.12 too, so without this the bundle
# leaves it out, and the runtime image has none: the slicer exits with 127.
bundle_args=()
shopt -s nullglob
for python_lib in slicer-src/deps/build/*/usr/local/libpython/lib/libpython3*.so.1.0; do
  bundle_args+=(--bundle-library "$(basename "$python_lib")")
done
shopt -u nullglob

python3 tools/stage_bundle.py \
  --executable slicer-src/build/src/Release/orca-slicer \
  --executable slicer-src/build/src/orca-slicer \
  --name orca-slicer \
  --arch "${ARCH:?ARCH is required}" \
  --output slicer-src/build/slicer_out \
  --resources slicer-src/resources \
  --library-root slicer-src/build/src \
  --library-root slicer-src/deps/build \
  "${bundle_args[@]}" \
  "${strip_args[@]}" \
  --json | tee slicer-src/build/slicer-bundle-report.json

if [[ -n "${SLICER_RELEASE_TAG:-}" ]]; then
  bash tools/verify_orca_version.sh \
    slicer-src/build/slicer_out/bin/orca-slicer \
    "$SLICER_RELEASE_TAG"
fi
