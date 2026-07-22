#!/usr/bin/env bash
set -euo pipefail

bash ./tools/run_config_dump.sh \
  --source slicer-src \
  --output slicer-out \
  --executable build/src/Release/bambu-studio \
  --executable build/src/bambu-studio
