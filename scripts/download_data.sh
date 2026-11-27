#!/usr/bin/env bash
# Download Wayfair's public WANDS dataset (~100 MB) into data/WANDS.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/WANDS ]; then
  git clone --depth 1 https://github.com/wayfair/WANDS.git data/WANDS
fi
ls -lh data/WANDS/dataset
