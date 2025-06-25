#!/bin/bash

pushd slicer-src
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./build/src/qidi-slicer
popd

cp ./slicer-src/build/src/*.json ./slicer-out
