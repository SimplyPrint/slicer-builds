#!/usr/bin/env bash

# Shared by the Cura build steps. Call cura_conan_env after a Conan executable
# is available; it exports CURA_CONAN_BIN and initializes the UltiMaker config.
_CURA_CONAN_CONFIG_URL_DEFAULT="https://github.com/Ultimaker/conan-config.git"
_CURA_CONAN_CONFIG_REF_DEFAULT="master"

_cura_resolve_conan_config_ref() {
  local config_url="$1"
  local config_ref="$2"
  local line name object_id key
  local -A candidates=()
  local -A peeled=()

  if [[ "$config_ref" =~ ^[0-9a-fA-F]{40,64}$ ]]; then
    printf '%s\n' "${config_ref,,}"
    return 0
  fi

  while IFS=$'\t' read -r object_id name; do
    [[ -n "$object_id" && -n "$name" ]] || continue
    if [[ ! "$object_id" =~ ^[0-9a-fA-F]{40,64}$ ]]; then
      echo "Cannot parse Conan config ref $config_ref from $config_url" >&2
      return 1
    fi
    if [[ "$name" == *'^{}' ]]; then
      key="${name%\^\{\}}"
      peeled["$key"]="${object_id,,}"
    else
      candidates["$name"]="${object_id,,}"
    fi
  done < <(git ls-remote --exit-code "$config_url" "$config_ref" "${config_ref}^{}")

  for key in "${!peeled[@]}"; do
    candidates["$key"]="${peeled[$key]}"
  done
  if (( ${#candidates[@]} != 1 )); then
    echo "Conan config ref $config_ref is missing or ambiguous at $config_url" >&2
    return 1
  fi
  for object_id in "${candidates[@]}"; do
    printf '%s\n' "$object_id"
  done
}

_cura_install_conan_config() {
  local config_url="$1"
  local config_ref="$2"
  local checkout archive temporary

  temporary="$(mktemp -d "${TMPDIR:-/tmp}/cura-conan-config.XXXXXX")"
  checkout="$temporary/repository"
  archive="$temporary/config.zip"

  # Conan's git installer accepts branches and tags, but an immutable commit
  # is not a valid `git clone --branch` value. Fetch the requested ref first
  # and give Conan a source archive so SHA, tag, and branch overrides all work.
  if ! git init --quiet "$checkout" \
    || ! git -C "$checkout" fetch --quiet --depth=1 "$config_url" "$config_ref" \
    || ! git -C "$checkout" archive \
      --format=zip --output="$archive" FETCH_HEAD \
    || ! "$CURA_CONAN_BIN" config install "$archive"; then
    rm -rf -- "$temporary"
    return 1
  fi

  rm -rf -- "$temporary"
}

cura_resolve_conan_config() {
  CURA_CONAN_CONFIG_URL="${CURA_CONAN_CONFIG_URL:-$_CURA_CONAN_CONFIG_URL_DEFAULT}"
  CURA_CONAN_CONFIG_REF="$(_cura_resolve_conan_config_ref \
    "$CURA_CONAN_CONFIG_URL" \
    "${CURA_CONAN_CONFIG_REF:-$_CURA_CONAN_CONFIG_REF_DEFAULT}")"
  export CURA_CONAN_CONFIG_URL CURA_CONAN_CONFIG_REF
}

cura_conan_env() {
  local config_identity config_ref config_stamp config_stamp_tmp config_url
  local deps default_conan installed_identity

  deps="$PWD/slicer-src/deps/build"
  default_conan="$deps/venv/bin/conan"
  CURA_CONAN_BIN="${CONAN:-$default_conan}"
  export CURA_CONAN_BIN
  export CONAN_HOME="${CONAN_HOME:-$deps/conan}"

  if [[ "$CURA_CONAN_BIN" == */* ]]; then
    if [[ ! -x "$CURA_CONAN_BIN" ]]; then
      echo "Conan executable not found: $CURA_CONAN_BIN" >&2
      return 1
    fi
  elif ! command -v "$CURA_CONAN_BIN" >/dev/null 2>&1; then
    echo "Conan executable not found on PATH: $CURA_CONAN_BIN" >&2
    return 1
  fi

  # Cura's recipes and Jinja profiles live in its public Conan config. Resolve
  # the configured branch/tag to a commit before caching or installing it.
  if [[ "${CURA_CONAN_CONFIG_INSTALL:-1}" != 0 ]]; then
    cura_resolve_conan_config
    config_url="$CURA_CONAN_CONFIG_URL"
    config_ref="$CURA_CONAN_CONFIG_REF"
    config_stamp="$CONAN_HOME/.slicer-builds-cura-conan-config"
    config_identity="$(
      printf '%s\0%s\0' "$config_url" "$config_ref" | sha256sum | cut -d' ' -f1
    )"
    installed_identity=""
    if [[ -f "$config_stamp" ]]; then
      IFS= read -r installed_identity < "$config_stamp" || true
    fi

    if [[ "$installed_identity" != "$config_identity" ]] \
      || ! grep -Eq '^core:default_profile[[:space:]]*=[[:space:]]*cura\.jinja' \
        "$CONAN_HOME/global.conf" 2>/dev/null; then
      _cura_install_conan_config "$config_url" "$config_ref"
      mkdir -p "$CONAN_HOME"
      config_stamp_tmp="$(mktemp "$CONAN_HOME/.cura-conan-config.XXXXXX")"
      printf '%s\n' "$config_identity" > "$config_stamp_tmp"
      mv -f -- "$config_stamp_tmp" "$config_stamp"
    fi

    # cura.jinja includes Conan's detected default profile. Refresh it on every
    # invocation because the shared cache can outlive a CC/CXX or builder
    # toolchain change; detection is cheap compared with a dependency build.
    "$CURA_CONAN_BIN" profile detect --force --name default
  fi
}
