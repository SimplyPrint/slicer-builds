#!/bin/bash
# https://github.com/prusa3d/PrusaSlicer/blob/master/doc/How%20to%20build%20-%20Linux%20et%20al.md


pushd slicer-src

# Build dependencies

cd deps
mkdir build
cd build
cmake .. -DDEP_WX_GTK3=ON
make
cd ../..

mkdir build
cd build
cmake .. -DSLIC3R_STATIC=1 -DSLIC3R_GTK=3 -DSLIC3R_PCH=OFF -DCMAKE_PREFIX_PATH=$(pwd)/../deps/build/destdir/usr/local
make -j$(nproc)

popd