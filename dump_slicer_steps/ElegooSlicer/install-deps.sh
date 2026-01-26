#!/bin/bash
set -euo pipefail

pushd slicer-src

sudo apt-get install -y libgtk-3-dev
if [[ -f "build_linux.sh" ]]; then
  ./build_linux.sh -u
else
  ./BuildLinux.sh -u
fi

popd