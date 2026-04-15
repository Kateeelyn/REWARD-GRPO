#!/usr/bin/env bash
# Creates a virtual environment and installs all GRPO pipeline dependencies.
set -e

VENV_DIR="$(dirname "$0")/venv"

echo "Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Activating and installing packages ..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$(dirname "$0")/requirements.txt"

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
