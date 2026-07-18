#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESOURCES="${1:?Usage: test-artifacts.sh <Cura resources directory>}"

output="$(mktemp -d)"
trap 'rm -rf "${output}"' EXIT

# shellcheck source=../version.env
source "${REPO_ROOT}/slicers/Cura/version.env"

python3 "${REPO_ROOT}/slicers/Cura/tools/generate-configs.py" \
  --resources "${RESOURCES}" \
  --output "${output}" \
  --version "${CURA_VERSION}" \
  --engine-ref "${CURA_ENGINE_REF}" \
  --resources-ref "${CURA_RESOURCES_REF}"

python3 "${SCRIPT_DIR}/test-artifacts.py" "${output}" "${RESOURCES}"
node "${SCRIPT_DIR}/validate-conditions.js" \
  "${output}/conditional_visibility.json" \
  "${output}/print_config_def.json"

for artifact in print_config_def.json machine.json filament.json process.json conditional_visibility.json metadata.json; do
  diff -u "${REPO_ROOT}/slicers/Cura/out/${CURA_VERSION}/${artifact}" "${output}/${artifact}"
done

echo "Cura generated artifacts are reproducible"
