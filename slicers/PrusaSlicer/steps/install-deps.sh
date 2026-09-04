#!/usr/bin/env bash
set -euo pipefail

packages=(
  autoconf
  build-essential
  cmake
  git
  libcurl4-openssl-dev
  libdbus-1-dev
  libglew-dev
  libglu1-mesa-dev
  libhidapi-dev
  libpng-dev
  ninja-build
  pkg-config
  texinfo
  zlib1g-dev
)

if [[ -d slicer-src/src/slic3r-app-launcher ]]; then
  gui=1
else
  case "${SLICER_GUI:-0}" in
    1 | true | TRUE | on | ON) gui=1 ;;
    *) gui=0 ;;
  esac
fi

case "$gui" in
  1)
    packages+=(libgtk-3-dev libwebkit2gtk-4.1-dev xauth xvfb)
    ;;
esac

if command -v sudo >/dev/null 2>&1; then
  apt=(sudo apt-get)
elif [[ $(id -u) == 0 ]]; then
  apt=(apt-get)
else
  echo "Installing dependencies requires sudo or a root shell." >&2
  exit 1
fi

"${apt[@]}" install -y --no-install-recommends "${packages[@]}"
