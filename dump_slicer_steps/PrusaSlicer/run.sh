#!/bin/bash

pushd slicer-src
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./prusa-slicer
popd

cp ./slicer-src/*.json ./slicer-out
