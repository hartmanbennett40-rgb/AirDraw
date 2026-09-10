#!/usr/bin/env bash
# Starts AirDraw: creates/uses a venv, installs deps if needed, runs main.py.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install -q -r requirements.txt

python -m airdraw.app
