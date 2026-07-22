#!/usr/bin/env bash
set -euo pipefail

binary="slicer-src/build/src/superslicer"
if [[ ! -x "$binary" ]]; then
  echo "SuperSlicer executable not found at $binary" >&2
  exit 1
fi

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run "$binary"

shopt -s nullglob
generated=(slicer-src/build/src/*.json)
if (( ${#generated[@]} == 0 )); then
  echo "SuperSlicer did not generate any JSON configuration files" >&2
  exit 1
fi
cp "${generated[@]}" slicer-out/
