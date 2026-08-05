#!/bin/bash
set -euo pipefail

pushd slicer-src
sudo bash ./BuildLinux.sh -u
popd
