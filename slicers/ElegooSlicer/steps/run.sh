#!/bin/bash
set -euo pipefail

if [[ -x "slicer-src/build/package/AppRun" ]]; then
  pushd slicer-src/build/package

  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./AppRun

  popd

  cp ./slicer-src/build/package/bin/*.json ./slicer-out
elif [[ -x "slicer-src/build/src/elegoo-slicer" ]]; then
  pushd slicer-src/build/src

  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./elegoo-slicer

  popd

  cp ./slicer-src/build/src/*.json ./slicer-out
else
  echo "Could not find an ElegooSlicer launcher in build/package or build/src"
  exit 1
fi
