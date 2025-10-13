#!/usr/bin/env bash

pushd slicer-src/build

mkdir -p slicer_out/resources
mkdir -p slicer_out/bin

cp -r resources/* slicer_out/resources

cp src/bambu-studio slicer_out/bin

cp src/*.so* slicer_out/bin

zip -r ../../slicer-out/binary.zip slicer_out/

popd