#!/usr/bin/env bash
set -euo pipefail

# Usage: ./build_slicer_binaries/apply_patches.sh <slicer> <version>
# Applies patches from:
#   build_slicer_binaries/<slicer>/patches/all/*.patch      (all builds)
#   build_slicer_binaries/<slicer>/patches/<version>/*.patch (exact version only)
#
# <version> is "nightly" for nightly builds, or the release tag (e.g. "v2.3.0")

SLICER="${1:?Usage: apply_patches.sh <slicer> <version>}"
VERSION="${2:?Usage: apply_patches.sh <slicer> <version>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_BASE="${SCRIPT_DIR}/${SLICER}/patches"
applied=0

apply_from_dir() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0

    for patch_file in $(find "$dir" -maxdepth 1 -name '*.patch' -type f | sort); do
        echo "Applying patch: ${patch_file}"
        git -C slicer-src apply "${patch_file}" --whitespace=fix
        ((++applied))
    done
}

apply_from_dir "${PATCHES_BASE}/all"
apply_from_dir "${PATCHES_BASE}/${VERSION}"

echo "Applied ${applied} patch(es)."
