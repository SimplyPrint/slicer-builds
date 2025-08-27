#!/bin/bash

pushd slicer-src

if [[ -f "build_linux.sh" ]]; then
  ./build_linux.sh -dr
else
  ./BuildLinux.sh -dr
fi

popd