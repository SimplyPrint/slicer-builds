#!/bin/bash
set -euo pipefail

pushd slicer-src

sudo ./BuildLinux.sh -u

popd
