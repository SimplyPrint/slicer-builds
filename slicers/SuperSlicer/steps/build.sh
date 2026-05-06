#!/bin/bash
set -euo pipefail


pushd slicer-src

./BuildLinux.sh -s

popd