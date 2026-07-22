#!/bin/bash
set -euo pipefail

bash ./tools/run_config_dump.sh \
  --source slicer-src \
  --output slicer-out \
  --executable build/src/Release/elegoo-slicer \
  --executable build/src/elegoo-slicer
