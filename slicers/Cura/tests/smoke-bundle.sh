#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUNDLE="$(realpath "${1:?Usage: smoke-bundle.sh <extracted-bundle>}")"
ENGINE="${BUNDLE}/bin/CuraEngine"
DEFINITION="${BUNDLE}/share/cura/resources/definitions/fdmprinter.def.json"
SETTING_VISIBILITY="${BUNDLE}/share/cura/resources/setting_visibility"
LICENSE_INVENTORY="${BUNDLE}/licenses/conan/inventory.json"

test -x "${ENGINE}"
test -f "${DEFINITION}"
test -d "${SETTING_VISIBILITY}"
find "${SETTING_VISIBILITY}" -maxdepth 1 -type f -name '*.cfg' -print -quit | grep -q .
test -f "${BUNDLE}/licenses/CuraEngine-AGPL-3.0.txt"
test -f "${BUNDLE}/licenses/Cura-LGPL-3.0.txt"
test -f "${LICENSE_INVENTORY}"

python3 - "${LICENSE_INVENTORY}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

inventory_path = Path(sys.argv[1]).resolve()
inventory = json.loads(inventory_path.read_text())
dependencies = inventory["dependencies"]
assert inventory["format_version"] == 1
assert inventory["generator"] == "simplyprint-conan-license-collector-v1"
assert inventory["dependency_count"] == len(dependencies) > 0
assert inventory["file_count"] == sum(item["file_count"] for item in dependencies)
assert len({item["destination"] for item in dependencies}) == len(dependencies)
for dependency in dependencies:
    destination = Path(dependency["destination"])
    assert len(destination.parts) == 1 and destination.name not in {"", ".", ".."}
    assert dependency["file_count"] == len(dependency["files"]) > 0
    for notice in dependency["files"]:
        relative = Path(notice["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert relative.parts[0] == destination.name
        bundled = (inventory_path.parent / relative).resolve()
        bundled.relative_to(inventory_path.parent)
        assert bundled.is_file()
        assert hashlib.sha256(bundled.read_bytes()).hexdigest() == notice["sha256"]
PY

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

ldd "${ENGINE}" >"${work_dir}/ldd.txt"
if grep -q 'not found' "${work_dir}/ldd.txt"; then
  cat "${work_dir}/ldd.txt" >&2
  exit 1
fi

# Validate every distributable shared object as well as the executable. Some
# Conan libraries are loaded dynamically and therefore do not appear in the
# engine's direct dependency graph exercised by ldd above.
for library in "${BUNDLE}"/lib/*; do
  [[ -e "${library}" ]] || continue
  ldd "${library}" >"${work_dir}/ldd-library.txt"
  if grep -q 'not found' "${work_dir}/ldd-library.txt"; then
    echo "Missing dependency for bundled library ${library}:" >&2
    cat "${work_dir}/ldd-library.txt" >&2
    exit 1
  fi
done

cp "${SCRIPT_DIR}/smoke.stl" "${work_dir}/smoke.stl"
python3 "${REPO_ROOT}/slicers/Cura/tools/generate-smoke-settings.py" \
  "${DEFINITION}" \
  "${work_dir}/resolved-settings.json"

if ! (
  cd "${work_dir}"
  "${ENGINE}" slice \
    -r resolved-settings.json \
    -o smoke.gcode \
    >engine.stdout \
    2>engine.stderr
); then
  cat "${work_dir}/engine.stdout" >&2
  cat "${work_dir}/engine.stderr" >&2
  exit 1
fi

test -s "${work_dir}/smoke.gcode"
grep -Eq '^;Generated with Cura_SteamEngine|^;FLAVOR:' "${work_dir}/smoke.gcode"
grep -q '^;LAYER_COUNT:' "${work_dir}/smoke.gcode"
grep -Eq '^G[01] ' "${work_dir}/smoke.gcode"

echo "Cura extracted-bundle smoke passed: $(wc -c < "${work_dir}/smoke.gcode") byte G-code"
