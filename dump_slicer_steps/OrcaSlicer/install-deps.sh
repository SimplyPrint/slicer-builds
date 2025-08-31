#!/bin/bash

pushd slicer-src

if [[ -f "build_linux.sh" ]]; then
  sudo ./build_linux.sh -u
else
  mv CMakeLists.txt CMakeLists.txt.prev
  cat <<EOF > CMakeLists.txt
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
EOF
  cat CMakeLists.txt.prev >> CMakeLists.txt
  rm CMakeLists.txt.prev
  sudo ./BuildLinux.sh -u
fi

popd