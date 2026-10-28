#!/bin/bash
set -euo pipefail

pushd slicer-src
# main still ships BuildLinux.sh; releases from 1.7.8 on ship build_linux.sh.
if [[ -f "build_linux.sh" ]]; then
  sudo bash ./build_linux.sh -u
else
  sudo bash ./BuildLinux.sh -u
fi
popd
