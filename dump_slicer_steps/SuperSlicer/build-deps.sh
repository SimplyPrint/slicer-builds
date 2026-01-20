#!/bin/bash
set -euo pipefail

# Build dependencies

pushd slicer-src

./BuildLinux.sh -d

popd