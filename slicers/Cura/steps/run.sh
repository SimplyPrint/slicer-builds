#!/usr/bin/env bash

set -euo pipefail

python3 slicers/Cura/tools/generate-configs.py \
  --resources slicer-src/resources \
  --output slicer-out
