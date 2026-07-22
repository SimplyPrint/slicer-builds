#!/usr/bin/env bash

set -euo pipefail

strip_args=()
[[ "${SLICER_STRIP:-1}" == 0 ]] || strip_args+=(--strip)

library_args=(--library-root slicer-src/build/Release)
for conan_root in slicer-src/deps/build/conan "${CONAN_HOME:-}"; do
  if [[ -n "$conan_root" && -d "$conan_root/p" ]]; then
    # Conan keeps installed package payloads in */p directories, alongside
    # build trees that may contain different copies of the same SONAME. Expose
    # only package payloads so staging cannot select an intermediate library.
    while IFS= read -r -d '' package_root; do
      library_args+=(--library-root "$package_root")
    done < <(
      find "$conan_root/p" -mindepth 2 -maxdepth 3 -type d -name p -print0 \
        | sort -z
    )
  fi
done

python3 tools/stage_bundle.py \
  --executable slicer-src/build/Release/CuraEngine \
  --executable slicer-src/build/CuraEngine \
  --name CuraEngine \
  --arch "${ARCH:?ARCH is required}" \
  --output slicer-src/build/slicer_out \
  "${library_args[@]}" \
  "${strip_args[@]}" \
  --json | tee slicer-src/build/slicer-bundle-report.json
