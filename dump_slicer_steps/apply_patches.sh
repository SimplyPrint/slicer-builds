#!/usr/bin/env bash
set -euo pipefail

SLICER="${1:?Usage: apply_patches.sh <slicer> <version>}"
VERSION="${2:?Usage: apply_patches.sh <slicer> <version>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_ROOT="${SCRIPT_DIR}/${SLICER}"
PATCHES_BASE="${PATCH_ROOT}/patches"
applied=0

apply_patch_file() {
    local patch_file="$1"
    [[ -f "$patch_file" ]] || return 0

    git -C slicer-src apply --check "$patch_file"
    git -C slicer-src apply "$patch_file" --whitespace=fix
    ((++applied))
}

apply_from_dir() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0

    for patch_file in $(find "$dir" -maxdepth 1 -name '*.patch' -type f | sort); do
        apply_patch_file "$patch_file"
    done
}

apply_from_dir "${PATCHES_BASE}/all"
apply_from_dir "${PATCHES_BASE}/${VERSION}"

if [[ "$applied" -eq 0 ]]; then
    apply_patch_file "${PATCH_ROOT}/dump_configs.patch"
fi

echo "Applied ${applied} patch(es)."
