#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:?Usage: stamp_version_date.sh <slicer-source-dir>}"
build_date="${SLICER_BUILD_DATE:-}"

if [[ -z "$build_date" ]]; then
  build_date="$(git -C "$source_dir" show -s --format=%cs HEAD)"
fi
if [[ ! "$build_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "SLICER_BUILD_DATE must use YYYY-MM-DD: $build_date" >&2
  exit 2
fi

version_file="$source_dir/version.inc"
[[ -f "$version_file" ]] || exit 0

# The upstream drivers replace +UNKNOWN with today's date. Stamping it first
# makes their substitution a no-op while retaining their behavior when this
# helper is not used.
if grep -Fq '+UNKNOWN' "$version_file"; then
  sed -i "s/+UNKNOWN/_${build_date}/g" -- "$version_file"
fi
