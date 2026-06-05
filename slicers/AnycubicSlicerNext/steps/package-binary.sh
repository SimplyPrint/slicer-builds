#!/usr/bin/env bash

set -euo pipefail

pushd slicer-src/build

mkdir -p slicer_out/resources
mkdir -p slicer_out/bin

cp -r resources/* slicer_out/resources

cp src/Release/orca-slicer slicer_out/bin

ldd src/Release/orca-slicer \
  | awk '/=> \// { print $3 }' \
  | sort -u \
  | xargs -r -I{} cp -L "{}" slicer_out/bin

popd
