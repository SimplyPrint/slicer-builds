#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: build_cmake_target.sh --source DIR --target TARGET --generator GENERATOR
                             [--config CONFIG] [--gettext SCRIPT]
                             [--auto-compiler-cache] -- [CMAKE_ARGS...]

Configure one source tree, build exactly one CMake target, and optionally run
the upstream gettext generator relative to the source directory.
EOF
  exit 2
}

source_dir=""
target=""
generator=""
config=""
gettext_script=""
auto_compiler_cache=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || usage
      source_dir="$2"
      shift 2
      ;;
    --target)
      [[ $# -ge 2 ]] || usage
      target="$2"
      shift 2
      ;;
    --generator)
      [[ $# -ge 2 ]] || usage
      generator="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || usage
      config="$2"
      shift 2
      ;;
    --gettext)
      [[ $# -ge 2 ]] || usage
      gettext_script="$2"
      shift 2
      ;;
    --auto-compiler-cache)
      auto_compiler_cache=1
      shift
      ;;
    --)
      shift
      break
      ;;
    *) usage ;;
  esac
done

[[ -n "$source_dir" && -n "$target" && -n "$generator" ]] || usage
[[ "$target" =~ ^[A-Za-z0-9_.+-]+$ ]] || usage
[[ -z "$config" || "$config" =~ ^[A-Za-z0-9_.+-]+$ ]] || usage
[[ -d "$source_dir" ]] || {
  echo "CMake source directory does not exist: $source_dir" >&2
  exit 2
}

source_dir="$(cd -- "$source_dir" && pwd -P)"
build_dir="$source_dir/build"

# CC/CXX, compile flags, linker flags, compiler launchers, and
# CMAKE_BUILD_PARALLEL_LEVEL intentionally remain environment-driven, matching
# the upstream wrappers while allowing slicerctl to control the toolchain.
compiler_cache_args=()
if [[ "$auto_compiler_cache" == 1 \
      && -z "${CMAKE_C_COMPILER_LAUNCHER:-}" \
      && -z "${CMAKE_CXX_COMPILER_LAUNCHER:-}" ]]; then
  compiler_cache="${CMAKE_CCACHE:-}"
  if [[ -n "$compiler_cache" ]]; then
    compiler_cache="$(command -v "$compiler_cache" || true)"
  elif command -v sccache >/dev/null 2>&1; then
    compiler_cache="$(command -v sccache)"
  elif command -v ccache >/dev/null 2>&1; then
    compiler_cache="$(command -v ccache)"
  fi
  if [[ -n "$compiler_cache" ]]; then
    compiler_cache_args+=(
      "-DCMAKE_C_COMPILER_LAUNCHER=$compiler_cache"
      "-DCMAKE_CXX_COMPILER_LAUNCHER=$compiler_cache"
    )
  fi
fi

cmake -S "$source_dir" -B "$build_dir" \
  "${compiler_cache_args[@]}" -G "$generator" "$@"

build_command=(cmake --build "$build_dir")
if [[ -n "$config" ]]; then
  build_command+=(--config "$config")
fi
build_command+=(--target "$target")
"${build_command[@]}"

if [[ -n "$gettext_script" ]]; then
  [[ "$gettext_script" != /* && "$gettext_script" != *".."* ]] || {
    echo "Gettext script must be relative to the source directory" >&2
    exit 2
  }
  [[ -f "$source_dir/$gettext_script" ]] || {
    echo "Gettext script does not exist: $source_dir/$gettext_script" >&2
    exit 2
  }
  (cd -- "$source_dir" && bash -- "./$gettext_script")
fi
