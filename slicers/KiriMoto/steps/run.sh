#!/usr/bin/env bash

set -euo pipefail

source_dir="${SLICER_SOURCE_DIR:-slicer-src}"

if [[ "$#" -eq 0 ]]; then
  exec node slicers/KiriMoto/tools/generate-config.mjs \
    --source "$source_dir" \
    --output slicer-out
fi

exec "$source_dir/build/slicer_out/bin/kirimoto" "$@"
