#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: build_slicer_pgo.sh <slicer> <family>" >&2
  exit 2
fi

slicer="$1"
family="$2"
[[ "$slicer" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
executable="$(
  python3 -c \
    'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["executable"])' \
    "slicers/$slicer/slicer.toml"
)"
[[ "$executable" =~ ^[A-Za-z0-9_.+-]+$ ]] || exit 2

build="./slicers/$slicer/steps/build.sh"
package="./slicers/$slicer/steps/package-binary.sh"
[[ -x "$build" && -x "$package" ]] || {
  echo "Missing build or package step for $slicer" >&2
  exit 2
}

if [[ "${SLICER_PGO:-1}" == 0 || "${ARCH:-}" != x86-64 ]]; then
  exec "$build"
fi
case "$family" in
  bambu | orca | prusa) ;;
  *) exec "$build" ;;
esac

base_cflags="${CFLAGS:-}"
base_cxxflags="${CXXFLAGS:-}"
base_ldflags="${LDFLAGS:-}"
profile_dir="$(mktemp -d "$PWD/slicer-src/.pgo-$slicer.XXXXXX")"
trap 'rm -rf -- "$profile_dir"' EXIT

echo "::group::Build instrumented $slicer"
generate_flags="-fprofile-generate=$profile_dir -fprofile-update=atomic -fprofile-reproducible=parallel-runs"
export CFLAGS="$base_cflags $generate_flags"
export CXXFLAGS="$base_cxxflags $generate_flags"
export LDFLAGS="$base_ldflags -fprofile-generate=$profile_dir"
"$build"
echo "::endgroup::"

echo "::group::Train $slicer PGO"
SLICER_STRIP=0 "$package"
python3 tools/train_slicer_pgo.py \
  --family "$family" \
  --source slicer-src \
  --bundle slicer-src/build/slicer_out \
  --executable "$executable"
profile_count="$(find "$profile_dir" -type f -name '*.gcda' -printf . | wc -c)"
if [[ "$profile_count" == 0 ]]; then
  echo "PGO training produced no GCC profiles" >&2
  exit 1
fi
echo "Collected $profile_count GCC profile files"
echo "::endgroup::"

echo "::group::Build optimized $slicer"
use_flags="-fprofile-use=$profile_dir -fprofile-correction -fprofile-partial-training -Wno-missing-profile"
export CFLAGS="$base_cflags $use_flags"
export CXXFLAGS="$base_cxxflags $use_flags"
export LDFLAGS="$base_ldflags -fprofile-use=$profile_dir"

# Environment flags only initialize a fresh CMake cache. Set them explicitly so
# the second pass recompiles the same build graph against the collected profile.
cmake -S slicer-src -B slicer-src/build \
  -DCMAKE_C_FLAGS:STRING="$CFLAGS" \
  -DCMAKE_CXX_FLAGS:STRING="$CXXFLAGS" \
  -DCMAKE_EXE_LINKER_FLAGS:STRING="$LDFLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS:STRING="$LDFLAGS" \
  -DCMAKE_MODULE_LINKER_FLAGS:STRING="$LDFLAGS"
"$build"
echo "::endgroup::"
