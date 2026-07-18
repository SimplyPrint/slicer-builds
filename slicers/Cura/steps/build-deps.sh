#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

prepare_pinned_sources
activate_conan

mapfile -t options < <(conan_options)
mkdir -p "$(dirname "${CURA_CONAN_GRAPH_FILE}")"
graph_tmp="$(mktemp "${CURA_CONAN_GRAPH_FILE}.tmp.XXXXXX")"
trap 'rm -f "${graph_tmp}"' EXIT
pushd "${CURA_ENGINE_SOURCE_DIR}"
conan install . \
  --settings build_type=Release \
  --build=missing \
  --format=json \
  --lockfile="${CURA_REPO_ROOT}/slicers/Cura/conan.lock" \
  "${options[@]}" \
  >"${graph_tmp}"
popd
python3 -m json.tool "${graph_tmp}" >/dev/null
mv "${graph_tmp}" "${CURA_CONAN_GRAPH_FILE}"
