#!/bin/bash
set -euo pipefail

pushd slicer-src/build/package

SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt xvfb-run ./orca-slicer &
pid=$!

expected_files=(
  "print_config_def.json"
  "process.json"
  "machine.json"
  "filament.json"
  "object_settings.json"
  "part_settings.json"
  "height_range_settings.json"
)

candidate_dirs=("." "./bin")

for _ in $(seq 1 60); do
  for dir in "${candidate_dirs[@]}"; do
    ready=true
    for file in "${expected_files[@]}"; do
      if [[ ! -f "$dir/$file" ]]; then
        ready=false
        break
      fi
    done

    if [[ "$ready" == true ]]; then
      out_dir="$dir"
      kill "$pid" || true
      wait "$pid" || true
      break 2
    fi
  done

  if ! kill -0 "$pid" 2>/dev/null; then
    wait "$pid"
    break
  fi

  sleep 1
done

if kill -0 "$pid" 2>/dev/null; then
  kill "$pid" || true
  wait "$pid" || true
fi

: "${out_dir:=}"

if [[ -z "$out_dir" ]]; then
  echo "Expected JSON files were not found in package root or bin directory"
  ls -la . || true
  ls -la ./bin || true
  exit 1
fi

popd

cp "./slicer-src/build/package/${out_dir#./}"/*.json ./slicer-out
