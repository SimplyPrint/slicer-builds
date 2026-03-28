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

for _ in $(seq 1 180); do
  ready=true
  for file in "${expected_files[@]}"; do
    if [[ ! -f "$file" ]]; then
      ready=false
      break
    fi
  done

  if [[ "$ready" == true ]]; then
    kill "$pid" || true
    wait "$pid" || true
    break
  fi

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

popd

cp ./slicer-src/build/package/*.json ./slicer-out
