#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_config_dump.sh --source DIR --output DIR --executable RELATIVE_PATH
                          [--executable RELATIVE_PATH ...]

Run the first available direct-build executable under Xvfb and collect the
JSON files emitted beside it by the config-dump patch.
EOF
  exit 2
}

source_dir=""
output_dir=""
executable_candidates=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || usage
      source_dir="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || usage
      output_dir="$2"
      shift 2
      ;;
    --executable)
      [[ $# -ge 2 ]] || usage
      executable_candidates+=("$2")
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ -n "$source_dir" && -n "$output_dir" ]] || usage
[[ ${#executable_candidates[@]} -gt 0 ]] || usage
[[ -d "$source_dir/resources" ]] || {
  echo "Slicer resource directory does not exist: $source_dir/resources" >&2
  exit 2
}

source_dir="$(cd -- "$source_dir" && pwd -P)"
output_dir="$(mkdir -p -- "$output_dir" && cd -- "$output_dir" && pwd -P)"
executable=""
for candidate in "${executable_candidates[@]}"; do
  [[ "$candidate" != /* && "$candidate" != *".."* ]] || usage
  if [[ -x "$source_dir/$candidate" ]]; then
    executable="$source_dir/$candidate"
    break
  fi
done
[[ -n "$executable" ]] || {
  echo "Could not find a config-dump executable under $source_dir" >&2
  exit 1
}

run_dir="$(cd -- "$(dirname -- "$executable")" && pwd -P)"
resource_parent="$(dirname -- "$run_dir")"
resource_link="$resource_parent/resources"
if [[ ! -e "$resource_link" && ! -L "$resource_link" ]]; then
  relative_resources="$(realpath --relative-to="$resource_parent" "$source_dir/resources")"
  ln -s -- "$relative_resources" "$resource_link"
fi

freshness_marker="$(mktemp --tmpdir="$run_dir" .slicer-config-dump.XXXXXX)"
trap 'rm -f -- "$freshness_marker"' EXIT
(
  cd -- "$run_dir"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    xvfb-run -- "$executable"
)

json_files=()
while IFS= read -r -d '' json_file; do
  json_files+=("$json_file")
done < <(
  find "$run_dir" -maxdepth 1 -type f -name '*.json' \
    -newer "$freshness_marker" -print0
)
[[ ${#json_files[@]} -gt 0 ]] || {
  echo "Config-dump executable did not create JSON files in $run_dir" >&2
  exit 1
}
for previous_json in "$output_dir"/*.json; do
  [[ -e "$previous_json" ]] || continue
  rm -f -- "$previous_json"
done
cp -- "${json_files[@]}" "$output_dir/"
