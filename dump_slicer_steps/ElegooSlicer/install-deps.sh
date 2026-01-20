#!/bin/bash
set -euo pipefail

pushd slicer-src

sudo apt-get install -y libgtk-3-dev
sudo ./BuildLinux.sh -u

popd