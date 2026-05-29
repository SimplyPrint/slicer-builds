#!/bin/bash
set -euo pipefail

pushd slicer-src

if [[ -f "build_linux.sh" ]]; then
  ./build_linux.sh -sir
else
  chmod +x BuildLinux.sh
  chmod +x run_gettext.sh
  ./BuildLinux.sh -sir
fi

popd
