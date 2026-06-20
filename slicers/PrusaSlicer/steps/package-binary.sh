#!/usr/bin/env bash

set -euo pipefail

pushd slicer-src/build

mkdir -p slicer_out/resources
mkdir -p slicer_out/bin

cp -r resources/* slicer_out/resources

cp src/prusa-slicer slicer_out/bin

ldd src/prusa-slicer \
  | awk '/=> \// { print $3 }' \
  | sort -u \
  | xargs -r -I{} cp -L "{}" slicer_out/bin

popd
