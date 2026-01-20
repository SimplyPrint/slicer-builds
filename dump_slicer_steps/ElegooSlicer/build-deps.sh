#!/bin/bash
set -euo pipefail

pushd slicer-src

./BuildLinux.sh -dr

popd