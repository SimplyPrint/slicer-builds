#!/bin/bash
set -euo pipefail

pushd slicer-src

if [[ -f "build_linux.sh" ]]; then
  bash ./build_linux.sh -dr
else
  bash ./BuildLinux.sh -dr
fi

popd
