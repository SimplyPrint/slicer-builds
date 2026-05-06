#!/usr/bin/env bash
set -euo pipefail

# Usage: ./tools/apply_versioned_patches.sh <slicer> <version> [dump|binary]
#
# Dump mode applies, in order:
#   slicers/<slicer>/patches/all/*.patch
#   slicers/<slicer>/patches/<version>/*.patch
#   slicers/<slicer>/patches/dump_configs.patch  (fallback when no versioned dump patch was applied)
#
# Binary mode applies, in order:
#   slicers/<slicer>/patches/binary/all/*.patch
#   slicers/<slicer>/patches/binary/<version>/*.patch

SLICER="${1:?Usage: apply_versioned_patches.sh <slicer> <version> [dump|binary]}"
VERSION="${2:?Usage: apply_versioned_patches.sh <slicer> <version> [dump|binary]}"
MODE="${3:-dump}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_ROOT="${REPO_ROOT}/slicers/${SLICER}/patches"
applied=0

apply_patch_file() {
    local patch_file="$1"
    [[ -f "$patch_file" ]] || return 0

    echo "Applying patch: ${patch_file}"
    git -C slicer-src apply --check "$patch_file"
    git -C slicer-src apply "$patch_file" --whitespace=fix
    ((++applied))
}

apply_from_dir() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0

    while IFS= read -r patch_file; do
        apply_patch_file "$patch_file"
    done < <(find "$dir" -maxdepth 1 -name '*.patch' -type f | sort)
}

case "$MODE" in
    dump)
        apply_from_dir "${PATCH_ROOT}/all"
        apply_from_dir "${PATCH_ROOT}/${VERSION}"
        if [[ "$applied" -eq 0 ]]; then
            apply_patch_file "${PATCH_ROOT}/dump_configs.patch"
        fi
        ;;
    binary)
        apply_from_dir "${PATCH_ROOT}/binary/all"
        apply_from_dir "${PATCH_ROOT}/binary/${VERSION}"
        ;;
    *)
        echo "Unknown patch mode: ${MODE}" >&2
        exit 1
        ;;
esac

echo "Applied ${applied} patch(es)."
