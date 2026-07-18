#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

prepare_pinned_sources

python3 "${CURA_REPO_ROOT}/slicers/Cura/tools/generate-configs.py" \
  --resources "${CURA_RESOURCES_SOURCE_DIR}/resources" \
  --output "${CURA_REPO_ROOT}/slicer-out" \
  --version "${CURA_VERSION}" \
  --engine-ref "${CURA_ENGINE_REF}" \
  --resources-ref "${CURA_RESOURCES_REF}"
