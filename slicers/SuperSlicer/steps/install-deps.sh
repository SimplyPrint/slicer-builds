#!/usr/bin/env bash
set -euo pipefail

packages=(
  build-essential
  cmake
  curl
  gettext
  git
  libcurl4-openssl-dev
  libdbus-1-dev
  libexpat1-dev
  libglew-dev
  libglu1-mesa-dev
  libjpeg-dev
  libpng-dev
  libssl-dev
  m4
  ninja-build
  pkg-config
  zlib1g-dev
)

case "${SLICER_GUI:-0}" in
  1 | true | TRUE | on | ON)
    packages+=(libgtk-3-dev libudev-dev)
    ;;
esac

sudo apt-get install -y --no-install-recommends "${packages[@]}"
