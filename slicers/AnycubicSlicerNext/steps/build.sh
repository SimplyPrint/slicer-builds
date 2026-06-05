#!/bin/bash
set -euo pipefail

pushd slicer-src

if [[ -f "build_linux.sh" ]]; then
  ./build_linux.sh -sir
else
  ./BuildLinux.sh -sir
fi

popd