#!/bin/bash
set -euo pipefail

pushd slicer-src/build/src

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./CrealityPrint

popd

cp ./slicer-src/build/src/*.json ./slicer-out