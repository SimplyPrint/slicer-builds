#!/bin/bash
set -euo pipefail

run_dir="slicer-src/build/package"
if [[ ! -x "$run_dir/orca-slicer" && -x "$run_dir/bin/orca-slicer" ]]; then
  run_dir="$run_dir/bin"
elif [[ ! -x "$run_dir/orca-slicer" && -x "slicer-src/build/src/Release/orca-slicer" ]]; then
  run_dir="slicer-src/build/src/Release"
elif [[ ! -x "$run_dir/orca-slicer" && -x "slicer-src/build/src/orca-slicer" ]]; then
  run_dir="slicer-src/build/src"
fi

pushd "$run_dir"

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./orca-slicer

popd

cp "$run_dir"/*.json ./slicer-out
