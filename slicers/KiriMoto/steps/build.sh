#!/usr/bin/env bash

set -euo pipefail

source_dir="${SLICER_SOURCE_DIR:-slicer-src}"
output_dir="${KIRIMOTO_BUILD_DIR:-$source_dir/build/kirimoto-node}"

node slicers/KiriMoto/tools/build-node-package.mjs \
  --source "$source_dir" \
  --output "$output_dir"
