#!/bin/bash
set -euo pipefail

pushd slicer-src
bash ./build_linux.sh -dr
popd
