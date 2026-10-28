#!/bin/bash
set -euo pipefail

# From 7.3.0 the deps build a private Pango runtime with meson against the
# system GLib, HarfBuzz, Cairo and Xft.
sudo apt-get install -y meson libglib2.0-dev libharfbuzz-dev libfribidi-dev \
  libcairo2-dev libfontconfig-dev libfreetype-dev libxft-dev libthai-dev

pushd slicer-src

if [[ -f "build_linux.sh" ]]; then
  sudo ./build_linux.sh -u
else
  chmod +x BuildLinux.sh
  sudo apt install bc -y
  sudo ./BuildLinux.sh -u
fi

popd