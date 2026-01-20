#!/bin/bash
set -euo pipefail
# https://github.com/bambulab/BambuStudio/wiki/Linux-Compile-Guide

pushd slicer-src

./BuildLinux.sh -sir

popd