#!/bin/bash
set -euo pipefail

# 1.5.3.5 copies the Agora SDK (libaosl.so, libagora_rtm_sdk.so) next to the
# executable without an rpath; its AppImage puts that directory on the path.
build_dirs="$PWD/slicer-src/build/src/Release:$PWD/slicer-src/build/src"
export LD_LIBRARY_PATH="$build_dirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

bash ./tools/run_config_dump.sh \
  --source slicer-src \
  --output slicer-out \
  --translation-domain ElegooSlicer \
  --executable build/src/Release/elegoo-slicer \
  --executable build/src/elegoo-slicer
