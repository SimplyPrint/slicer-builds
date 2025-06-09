#!/bin/bash

pushd slicer-src/build/src

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./elegoo-slicer

popd

cp ./slicer-src/build/src/*.json ./slicer-out