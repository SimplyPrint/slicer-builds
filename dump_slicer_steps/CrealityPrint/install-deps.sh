#!/bin/bash
set -euo pipefail

pushd slicer-src

if [[ -f "build_linux.sh" ]]; then
  sudo ./build_linux.sh -u
else
  chmod +x BuildLinux.sh
  sudo apt install bc -y
  sudo ./BuildLinux.sh -u
fi

popd