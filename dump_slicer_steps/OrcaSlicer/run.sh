#!/bin/bash

pushd slicer-src/build/package

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./orca-slicer

popd

cp ./slicer-src/build/package/bin/*.json ./slicer-out