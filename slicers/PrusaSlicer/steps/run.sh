#!/bin/bash
set -euo pipefail

pushd slicer-src
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./build/src/prusa-slicer
popd

cp ./slicer-src/build/src/*.json ./slicer-out

python3 ./tools/annotate_editor_semantics.py \
  --output ./slicer-out \
  --overlay ./slicers/PrusaSlicer/semantics.toml \
  --require-tool-reference
