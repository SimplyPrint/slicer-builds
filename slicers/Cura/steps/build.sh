#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

prepare_pinned_sources
activate_conan

mapfile -t options < <(conan_options)
pushd "${CURA_ENGINE_SOURCE_DIR}"
conan build . \
  --settings build_type=Release \
  "${options[@]}"
popd

test -x "${CURA_ENGINE_SOURCE_DIR}/build/Release/CuraEngine"
