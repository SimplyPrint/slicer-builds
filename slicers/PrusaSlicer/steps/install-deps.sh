#!/bin/bash
set -euo pipefail

sudo apt-get install -y \
    git \
    build-essential \
    autoconf \
    cmake \
    libglu1-mesa-dev \
    libgtk-3-dev \
    libdbus-1-dev \
    libhidapi-dev \
    libwebkit2gtk-4.1-dev \
    texinfo
