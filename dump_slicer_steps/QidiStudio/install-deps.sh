#!/bin/bash
set -euo pipefail
# https://github.com/bambulab/BambuStudio/wiki/Linux-Compile-Guide

# Install numpy for OpenCV build
sudo apt-get install -y python3-numpy

pushd slicer-src

sudo ./BuildLinux.sh -u

popd