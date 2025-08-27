#!/bin/bash

pushd slicer-src

if [[ -f "build_linux.sh" ]]; then
  sudo ./build_linux.sh -u
else
  sudo ./BuildLinux.sh -u
fi

popd