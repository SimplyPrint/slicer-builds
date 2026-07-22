#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: cache-fingerprint.sh <deps|build> <source-dir> <slicer> <version> <binary|dump>

Print a deterministic SHA-256 fingerprint for one slicer cache. The source
checkout must already have its version-specific patches applied.
EOF
  exit 2
}

[[ $# == 5 ]] || usage

scope="$1"
source_dir="$2"
slicer="$3"
version="$4"
patch_mode="$5"

case "$scope" in
  deps | build) ;;
  *) usage ;;
esac
case "$patch_mode" in
  binary | dump) ;;
  *) usage ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
source_dir="$(cd "$source_dir" && pwd)"
slicer_dir="$repo_root/slicers/$slicer"

git -C "$source_dir" rev-parse --git-dir >/dev/null 2>&1 || {
  echo "Not a Git checkout: $source_dir" >&2
  exit 2
}
[[ -d "$slicer_dir" ]] || {
  echo "Unknown slicer: $slicer" >&2
  exit 2
}

fingerprint_input="$(mktemp)"
patch_list="${fingerprint_input}.patches"
trap 'rm -f "$fingerprint_input" "$patch_list"' EXIT

append_literal() {
  local label="$1"
  local value="$2"
  printf 'literal\0%s\0%s\0' "$label" "$value" >> "$fingerprint_input"
}

append_file() {
  local label="$1"
  local path="$2"
  local mode digest target

  if [[ -L "$path" ]]; then
    target="$(readlink -- "$path")"
    printf 'symlink\0%s\0%s\0' "$label" "$target" >> "$fingerprint_input"
    return
  fi

  [[ -f "$path" ]] || return 0
  mode="$(stat -c '%a' -- "$path")"
  digest="$(sha256sum -- "$path" | cut -d' ' -f1)"
  printf 'file\0%s\0%s\0%s\0' "$label" "$mode" "$digest" >> "$fingerprint_input"
}

is_dependency_input() {
  local path="$1"
  case "$path" in
    deps/* | deps_src/* | dependencies/* | cmake/* | \
      linux.d/* | scripts/linux.d/* | \
      CMakeLists.txt | CMakePresets.json | CMakeUserPresets.json | \
      conandata.yml | conanfile.py | conanfile.txt | \
      vcpkg.json | vcpkg-configuration.json | \
      pyproject.toml | poetry.lock | requirements*.txt | \
      BuildLinux.sh | build_linux.sh | build_release_linux.sh | \
      scripts/build* | scripts/*deps*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

append_upstream_inputs() {
  local record path metadata mode

  if [[ "$scope" == build ]]; then
    # A tree ID covers all tracked sources/resources and submodule gitlinks.
    # Applied patches are fingerprinted separately below.
    append_literal upstream.tree "$(git -C "$source_dir" rev-parse 'HEAD^{tree}')"
    return
  fi

  # Keep third-party dependency caches reusable across source-only commits.
  # Enumerate tracked inputs so generated DL_CACHE/build content is ignored,
  # but hash the working-tree bytes after deterministic install/patch steps.
  while IFS= read -r -d '' record; do
    path="${record#*$'\t'}"
    is_dependency_input "$path" || continue
    metadata="${record%%$'\t'*}"
    mode="${metadata%% *}"
    if [[ "$mode" == "160000" ]]; then
      printf 'literal\0upstream/%s\0%s\0' "$path" "$metadata" >> "$fingerprint_input"
    elif [[ -e "$source_dir/$path" || -L "$source_dir/$path" ]]; then
      append_file "upstream/$path" "$source_dir/$path"
    else
      printf 'literal\0upstream/%s\0missing\0' "$path" >> "$fingerprint_input"
    fi
  done < <(git -C "$source_dir" ls-files --stage -z)
}

append_repository_input() {
  local label="$1"
  local relative_path="$2"
  append_file "repository/$label" "$repo_root/$relative_path"
}

append_patch_inputs() {
  local patch_file patch_digest
  local patch_index=0

  patch_affects_dependencies() {
    local patch_path="$1"
    local marker git_word old_path new_path path
    local saw_diff=false

    while IFS=' ' read -r marker git_word old_path new_path; do
      [[ "$marker" == diff && "$git_word" == --git ]] || continue
      saw_diff=true
      for path in "$old_path" "$new_path"; do
        path="${path#\"}"
        path="${path%\"}"
        path="${path#a/}"
        path="${path#b/}"
        if is_dependency_input "$path"; then
          return 0
        fi
      done
    done < "$patch_path"

    # Conservatively include patches without standard Git headers.
    [[ "$saw_diff" == false ]]
  }

  append_patch_if_relevant() {
    local patch_path="$1"

    [[ -f "$patch_path" ]] || return 0
    if [[ "$scope" == deps ]] && ! patch_affects_dependencies "$patch_path"; then
      return 0
    fi

    # Ordered patch content affects the output; a version-alias rename does not.
    ((++patch_index))
    patch_digest="$(sha256sum -- "$patch_path" | cut -d' ' -f1)"
    printf 'patch\0%08d\0%s\0' "$patch_index" "$patch_digest" >> "$fingerprint_input"
  }

  python3 "$repo_root/tools/slicerctl.py" \
    patch-files "$slicer" "$version" "$patch_mode" --null > "$patch_list"
  while IFS= read -r -d '' patch_file; do
    append_patch_if_relevant "$repo_root/$patch_file"
  done < "$patch_list"
}

append_toolchain_inputs() {
  local command_name command_path command_version package_versions
  local -a commands=(cc c++ gcc g++ clang clang++ ld cmake ninja make python3 conan git ccache sccache)

  append_literal context.arch "${ARCH:-${RUNNER_ARCH:-$(uname -m)}}"
  append_literal context.os "${ImageOS:-unknown}"
  append_literal context.image_version "${ImageVersion:-unknown}"
  append_literal context.slicer_gui "${SLICER_GUI:-0}"
  append_literal context.cmake_generator "${CMAKE_GENERATOR:-default}"
  append_literal context.cc "${CC:-default}"
  append_literal context.cxx "${CXX:-default}"
  append_literal context.cflags "${CFLAGS:-}"
  append_literal context.cxxflags "${CXXFLAGS:-}"
  append_literal context.ldflags "${LDFLAGS:-}"
  append_literal context.slicer_pch "${SLICER_PCH:-default}"
  if [[ "$scope" == build ]]; then
    # Release tags affect embedded slicer metadata, not third-party dependencies.
    append_literal context.slicer_release_tag "${SLICER_RELEASE_TAG:-}"
    case "$slicer" in
      AnycubicSlicerNext | OrcaSlicer)
        append_literal context.orca_extra_build_args "${ORCA_EXTRA_BUILD_ARGS:-}"
        append_literal context.orca_updater_sig_key "${ORCA_UPDATER_SIG_KEY:-}"
        ;;
      ElegooSlicer)
        append_literal context.elegoo_extra_build_args "${ELEGOO_EXTRA_BUILD_ARGS:-}"
        append_literal context.orca_updater_sig_key "${ORCA_UPDATER_SIG_KEY:-}"
        ;;
    esac
  fi
  append_literal context.conan "${CONAN:-default}"
  append_literal context.conan_home "${CONAN_HOME:-default}"
  if [[ "$slicer" == "Cura" ]]; then
    append_literal context.cura_conan_config_install "${CURA_CONAN_CONFIG_INSTALL:-default}"
    append_literal context.cura_conan_config_ref "${CURA_CONAN_CONFIG_REF:-default}"
    append_literal context.cura_conan_config_url "${CURA_CONAN_CONFIG_URL:-default}"
  fi
  append_literal context.glibc "$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"

  for command_name in "${commands[@]}"; do
    command_path="$(command -v "$command_name" 2>/dev/null || true)"
    [[ -n "$command_path" ]] || continue
    command_version="$("$command_path" --version 2>&1 || true)"
    append_literal "tool/$command_name" "$command_path
$command_version"
  done

  # Cura's pinned Conan is normally in a source-local virtual environment.
  for command_path in \
    "${CONAN:-}" \
    "$source_dir/deps/build/venv/bin/conan" \
    "$source_dir/deps/conan/bin/conan" \
    "$source_dir/.venv/bin/conan" \
    /opt/conan/bin/conan; do
    [[ -x "$command_path" ]] || continue
    command_version="$("$command_path" --version 2>&1 || true)"
    append_literal "tool/conan:$command_path" "$command_version"
  done

  if command -v dpkg-query >/dev/null 2>&1; then
    package_versions="$({
      dpkg-query -W -f='${binary:Package}=${Version}\n' 2>/dev/null || true
    } | awk -F= '
      $1 ~ /-dev(:[^=]+)?$/ ||
      $1 ~ /^(build-essential|ccache|clang|cmake|g\+\+|gcc|libc6|lld|make|ninja-build|pkg-config|sccache)(:[^=]+)?$/
    ' | LC_ALL=C sort)"
    append_literal tool.debian_build_packages "$package_versions"
  fi
}

append_upstream_inputs

append_repository_input "cache-fingerprint-tool" ".github/scripts/cache-fingerprint.sh"
append_repository_input "patch-selector" "tools/apply_versioned_patches.sh"
append_repository_input "install-deps-step" "slicers/$slicer/steps/install-deps.sh"
append_repository_input "build-deps-step" "slicers/$slicer/steps/build-deps.sh"
append_repository_input "conan-env-step" "slicers/$slicer/steps/conan-env.sh"
if [[ "$scope" == build ]]; then
  append_repository_input "build-step" "slicers/$slicer/steps/build.sh"
  if grep -Fq 'tools/stamp_version_date.sh' "$slicer_dir/steps/build.sh"; then
    append_repository_input "stamp-version-date-tool" "tools/stamp_version_date.sh"
  fi
  if grep -Fq 'tools/build_cmake_target.sh' "$slicer_dir/steps/build.sh"; then
    append_repository_input "build-cmake-target-tool" "tools/build_cmake_target.sh"
  fi
fi
append_patch_inputs
append_toolchain_inputs

if [[ "$scope" == build ]]; then
  append_literal cache.dependencies "${CACHE_DEPS_FINGERPRINT:?CACHE_DEPS_FINGERPRINT is required for build fingerprints}"
fi

sha256sum "$fingerprint_input" | cut -d' ' -f1
