#!/bin/bash
set -euo pipefail

pushd slicer-src
bash ./BuildLinux.sh -dr
popd
