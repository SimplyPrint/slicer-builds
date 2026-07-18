#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

prepare_pinned_sources

build_dir="${CURA_ENGINE_SOURCE_DIR}/build"
binary="${build_dir}/Release/CuraEngine"
bundle="${build_dir}/slicer_out"

test -x "${binary}"
rm -rf "${bundle:?}"/*
mkdir -p \
  "${bundle}/bin" \
  "${bundle}/lib" \
  "${bundle}/licenses" \
  "${bundle}/share/cura/resources" \
  "${bundle}/share/simplyprint"

install -m 0755 "${binary}" "${bundle}/bin/CuraEngine"

# Keep the profile graph paired with the engine. The runtime receives fully
# resolved -r settings, but the definitions are useful for smoke tests and for
# diagnosing profile-resolution problems from the exact released resources.
for resource_dir in definitions extruders intent quality quality_changes variants; do
  if [[ -d "${CURA_RESOURCES_SOURCE_DIR}/resources/${resource_dir}" ]]; then
    mkdir -p "${bundle}/share/cura/resources/${resource_dir}"
    cp -a \
      "${CURA_RESOURCES_SOURCE_DIR}/resources/${resource_dir}/." \
      "${bundle}/share/cura/resources/${resource_dir}/"
  fi
done

install -m 0644 "${CURA_ENGINE_SOURCE_DIR}/LICENSE" "${bundle}/licenses/CuraEngine-AGPL-3.0.txt"
install -m 0644 "${CURA_RESOURCES_SOURCE_DIR}/LICENSE" "${bundle}/licenses/Cura-LGPL-3.0.txt"
python3 "${CURA_REPO_ROOT}/slicers/Cura/tools/collect_conan_licenses.py" \
  --graph "${CURA_CONAN_GRAPH_FILE}" \
  --output "${bundle}/licenses/conan"

# Conan options make nearly all third-party dependencies static. Copy any
# remaining non-system shared objects and teach the executable to look beside
# itself so the extracted zip does not depend on the build directory.
while IFS= read -r library; do
  [[ -n "${library}" ]] || continue
  case "${library}" in
    /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*) continue ;;
  esac
  cp -L "${library}" "${bundle}/lib/"
done < <(ldd "${binary}" | awk '/=> \// { print $3 }' | sort -u)

# The executable's direct dependency graph cannot reveal libraries loaded with
# dlopen(). Inspect every distributable host package selected by Conan's graph
# traits and preserve both its concrete versioned payload and ELF SONAME. This
# remains data-driven when upstream changes its dependency set.
python3 "${CURA_REPO_ROOT}/slicers/Cura/tools/collect_conan_runtime_libs.py" \
  --graph "${CURA_CONAN_GRAPH_FILE}" \
  --output "${bundle}/lib"

# Conan-built shared objects may retain a RUNPATH into the hashed package
# cache. Normalize every copied ELF to resolve transitive dependencies from
# the bundle's lib directory as well; otherwise the build host can mask a
# broken extracted bundle.
for library in "${bundle}"/lib/*; do
  if patchelf --print-soname "${library}" >/dev/null 2>&1; then
    # shellcheck disable=SC2016 # ELF loaders, not this shell, expand $ORIGIN.
    patchelf --force-rpath --set-rpath '$ORIGIN' "${library}"
  fi
done

# shellcheck disable=SC2016 # ELF loaders, not this shell, expand $ORIGIN.
patchelf --force-rpath --set-rpath '$ORIGIN/../lib' "${bundle}/bin/CuraEngine"

python3 - "${bundle}/share/simplyprint/build.json" <<PY
import json
import sys

metadata = {
    "arch": "${ARCH:-x86-64}",
    "engine_ref": "${CURA_ENGINE_REF}",
    "engine_repo": "${CURA_ENGINE_REPO}",
    "executable": "bin/CuraEngine",
    "platform": "${PLATFORM:-linux}",
    "resources_ref": "${CURA_RESOURCES_REF}",
    "resources_repo": "${CURA_RESOURCES_REPO}",
    "settings_contract": "cura-resolved-v1",
    "slicer": "Cura",
    "version": "${CURA_VERSION}",
}
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(metadata, output, indent=2, sort_keys=True)
    output.write("\\n")
PY

"${CURA_REPO_ROOT}/slicers/Cura/tests/smoke-bundle.sh" "${bundle}"
