#!/usr/bin/env bash

set -euo pipefail

required_node_major=22
current_node_major="$(node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/' || true)"

if [[ ! "$current_node_major" =~ ^[0-9]+$ ]] || (( current_node_major < required_node_major )); then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

node --version
npm --version
