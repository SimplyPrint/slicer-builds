#!/usr/bin/env bash

set -euo pipefail

binary="slicer-src/build/Release/CuraEngine"
bundle="slicer-src/build/slicer_out"
mkdir -p "$bundle/bin"
cp "$binary" "$bundle/bin/CuraEngine"

find slicer-src/deps/build/conan/p \
  \( -type f -o -type l \) -path '*/p/lib/*.so*' \
  -exec cp -P {} "$bundle/bin/" \;
