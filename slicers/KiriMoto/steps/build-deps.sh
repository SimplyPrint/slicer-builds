#!/usr/bin/env bash

set -euo pipefail

source_dir="${SLICER_SOURCE_DIR:-slicer-src}"
npm --prefix "$source_dir" install --ignore-scripts --no-audit --no-fund
npm --prefix "$source_dir" run webpack-three
npm --prefix "$source_dir" run webpack-zip
npm --prefix "$source_dir" run webpack-qjs
