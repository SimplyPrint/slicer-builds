#!/usr/bin/env bash

pushd slicer-src/build

mkdir -p slicer_out/resources
mkdir -p slicer_out/bin

cp -r resources/* slicer_out/resources

cp superslicer slicer_out/bin

popd