#!/bin/bash
set -euo pipefail

if [[ -x slicer-src/build/src/slic3r-app-launcher/slic3r-app-launcher ]]; then
  launcher="$PWD/slicer-src/build/src/slic3r-app-launcher/slic3r-app-launcher"
  model="$PWD/slicer-src/config-model.json"
  data_dir="$PWD/slicer-src/.simplyprint-config-dump"
  mkdir -p "$data_dir"
  pushd slicer-src
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run "$launcher" \
    --datadir "$data_dir" \
    --export-config-schema "$model"
  popd
  python3 tools/convert_prusa_3_config_model.py "$model" slicer-out
  cp slicer-src/config-model.json slicer-src/profile-bundle.json slicer-out/
else
  pushd slicer-src
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./build/src/prusa-slicer
  popd
  cp ./slicer-src/build/src/*.json ./slicer-out
fi
