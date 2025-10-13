#!/bin/bash

pushd slicer-src
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./build/bin/superslicer
popd

cp ./slicer-src/build/bin/*.json ./slicer-out
