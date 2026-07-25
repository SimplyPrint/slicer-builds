#!/bin/bash
set -euo pipefail

bash ./tools/run_config_dump.sh \
  --source slicer-src \
  --output slicer-out \
  --executable build/src/Release/FlashStudio \
  --executable build/src/FlashStudio \
  --executable "build/src/Release/Flash Studio" \
  --executable "build/src/Flash Studio"
