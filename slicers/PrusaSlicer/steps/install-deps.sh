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

case "${SLICER_GUI:-0}" in
  1 | true | TRUE | on | ON)
    packages+=(libgtk-3-dev libwebkit2gtk-4.1-dev)
    ;;
esac

sudo apt-get install -y --no-install-recommends "${packages[@]}"
