#!/bin/bash

pushd slicer-src

sudo apt install libgtk-3-dev
sudo ./BuildLinux.sh -u

popd