#!/bin/bash
set -euo pipefail
# https://github.com/bambulab/BambuStudio/wiki/Linux-Compile-Guide

# Install numpy for OpenCV build
sudo apt-get install -y python3-numpy

pushd slicer-src

sed -i "s/-DQDT_RELEASE_TO_PUBLIC=1/-DQDT_RELEASE_TO_PUBLIC=0/" BuildLinux.sh 
chmod +x ./BuildLinux.sh
sudo ./BuildLinux.sh -u

popd