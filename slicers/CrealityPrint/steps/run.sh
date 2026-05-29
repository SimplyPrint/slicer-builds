#!/bin/bash
set -euo pipefail

run_dir="slicer-src/build/src"
if [[ ! -x "$run_dir/CrealityPrint" && -x "$run_dir/Release/CrealityPrint" ]]; then
  run_dir="$run_dir/Release"
fi

pushd "$run_dir"

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./CrealityPrint

popd

cp "$run_dir"/*.json ./slicer-out
