#!/bin/bash
set -euo pipefail

run_dir="slicer-src/build/src"
if [[ ! -x "$run_dir/CrealityPrint" && -x "$run_dir/Release/CrealityPrint" ]]; then
  run_dir="$run_dir/Release"
fi

pushd "$run_dir"

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./CrealityPrint

popd

expected_json=(
  print_config_def.json
  process.json
  filament.json
  machine.json
  object_settings.json
  part_settings.json
  height_range_settings.json
)

for json_file in "${expected_json[@]}"; do
  if [[ ! -f "$run_dir/$json_file" ]]; then
    echo "Missing expected extracted JSON file: $json_file" >&2
    find "$run_dir" -maxdepth 1 -name '*.json' -print | sort >&2
    exit 1
  fi
done

cp "$run_dir"/*.json ./slicer-out
