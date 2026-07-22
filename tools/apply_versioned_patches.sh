#!/usr/bin/env bash
set -euo pipefail

# Usage: ./tools/apply_versioned_patches.sh <slicer> <version> [dump|binary]
#
# Patch order is selected by the slicer's manifest: declared shared patch sets
# first, then the slicer's local all/version overlays.

SLICER="${1:?Usage: apply_versioned_patches.sh <slicer> <version> [dump|binary]}"
VERSION="${2:?Usage: apply_versioned_patches.sh <slicer> <version> [dump|binary]}"
MODE="${3:-dump}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
applied=0
patch_list="$(mktemp)"
trap 'rm -f "$patch_list"' EXIT

apply_patch_file() {
    local patch_file="$1"
    [[ -f "$patch_file" ]] || return 0

    echo "Applying patch: ${patch_file}"
    git -C slicer-src apply --check "$patch_file"
    git -C slicer-src apply "$patch_file" --whitespace=fix
    ((++applied))
}

case "$MODE" in
    dump | binary) ;;
    *)
        echo "Unknown patch mode: ${MODE}" >&2
        exit 1
        ;;
esac

python3 "${REPO_ROOT}/tools/slicerctl.py" \
    patch-files "$SLICER" "$VERSION" "$MODE" --null > "$patch_list"

while IFS= read -r -d '' patch_file; do
    apply_patch_file "${REPO_ROOT}/${patch_file}"
done < "$patch_list"

echo "Applied ${applied} patch(es)."
