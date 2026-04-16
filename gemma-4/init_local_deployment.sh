#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d ~/.cache/huggingface ]; then
    mkdir -p ~/.cache/huggingface
fi
sudo chown -R "$USER:$USER" ~/.cache
chmod u+w ~/.cache
docker compose up --remove-orphans --build -d
