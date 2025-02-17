#!/bin/bash

pushd slicer-src
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./build/prusa-slicer
popd

cp ./slicer-src/build/*.json ./slicer-out
