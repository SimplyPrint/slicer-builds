#!/bin/bash
# https://github.com/prusa3d/PrusaSlicer/blob/master/doc/How%20to%20build%20-%20Linux%20et%20al.md

# Build dependencies

pushd slicer-src

cd deps
mkdir build
cd build
cmake .. -DDEP_WX_GTK3=ON
make
cd ../..

popd