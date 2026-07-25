#!/bin/bash
set -euo pipefail

bash ./tools/run_config_dump.sh \
  --source slicer-src \
  --output slicer-out \
  --translation-domain CrealityPrint \
  --executable build/src/CrealityPrint \
  --executable build/src/Release/CrealityPrint
