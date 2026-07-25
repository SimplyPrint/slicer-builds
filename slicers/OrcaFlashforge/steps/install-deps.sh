#!/bin/bash
set -euo pipefail

pushd slicer-src
sudo bash ./build_linux.sh -u
popd
