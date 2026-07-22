#!/usr/bin/env bash
set -euo pipefail

executable="${1:?Usage: verify_orca_version.sh <orca-executable> <release-tag>}"
release_tag="${2:?Usage: verify_orca_version.sh <orca-executable> <release-tag>}"

[[ -x "$executable" ]] || {
  echo "OrcaSlicer executable is not runnable: $executable" >&2
  exit 2
}
executable="$(realpath "$executable")"

expected_version="${release_tag#refs/tags/}"
expected_version="${expected_version#v}"
if [[ ! "$expected_version" =~ ^[0-9]+(\.[0-9]+){2}([-+._][0-9A-Za-z][0-9A-Za-z.+_-]*)?$ ]]; then
  echo "Unsupported OrcaSlicer release tag: $release_tag" >&2
  exit 2
fi

probe_dir="$(mktemp -d)"
output_file="$probe_dir/output"
trap 'rm -rf "$probe_dir"' EXIT

set +e
(
  cd "$probe_dir"
  timeout "${SLICER_VERSION_CHECK_TIMEOUT:-60}" "$executable" --help
) >"$output_file" 2>&1
status=$?
set -e
if [[ "$status" != 0 ]]; then
  echo "OrcaSlicer version probe exited with status $status:" >&2
  tail -40 "$output_file" >&2
  exit 1
fi

version_line="$(grep -m1 -E '^OrcaSlicer-[^:[:space:]]+:[[:space:]]*$' "$output_file" || true)"
actual_version="$(sed -nE 's/^OrcaSlicer-([^:[:space:]]+):[[:space:]]*$/\1/p' <<< "$version_line")"
if [[ -z "$actual_version" ]]; then
  echo "OrcaSlicer did not report its version in --help output:" >&2
  tail -40 "$output_file" >&2
  exit 1
fi
if [[ "$actual_version" != "$expected_version" ]]; then
  echo "OrcaSlicer version mismatch: release $release_tag contains $actual_version" >&2
  exit 1
fi

echo "Verified OrcaSlicer release version: $actual_version"
