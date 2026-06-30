#!/usr/bin/env bash

set -euo pipefail

pushd slicer-src/build

mkdir -p slicer_out/resources
mkdir -p slicer_out/bin

cp -r resources/* slicer_out/resources

binary_path="src/Release/orca-slicer"
if [[ ! -x "$binary_path" && -x "src/orca-slicer" ]]; then
  binary_path="src/orca-slicer"
fi

cp "$binary_path" slicer_out/bin

ldd "$binary_path" \
  | awk '/=> \// { print $3 }' \
  | sort -u \
  | xargs -r -I{} cp -L "{}" slicer_out/bin

popd
