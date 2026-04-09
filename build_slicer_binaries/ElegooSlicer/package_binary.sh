#!/usr/bin/env bash

set -euo pipefail

pushd slicer-src/build

mkdir -p slicer_out/resources
mkdir -p slicer_out/bin

if [[ -d "package/bin" ]]; then
  cp -r resources/* slicer_out/resources
  cp package/bin/* slicer_out/bin
elif [[ -x "src/elegoo-slicer" ]]; then
  cp -r resources/* slicer_out/resources
  cp src/elegoo-slicer slicer_out/bin
else
  echo "Could not find an ElegooSlicer binary in build/package/bin or build/src"
  exit 1
fi

popd
