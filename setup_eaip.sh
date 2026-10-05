#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
EAIP_ZIP="${EAIP_PROJECT_ZIP:-$(cd "$ROOT_DIR/../.." && pwd)/EAIP ASD project.zip}"
EAIP_CACHE="$BACKEND_DIR/.cache/eaip_project"

if [[ ! -f "$EAIP_ZIP" ]]; then
  echo "EAIP-DARV project ZIP not found:"
  echo "  $EAIP_ZIP"
  echo "Set EAIP_PROJECT_ZIP to its absolute path and rerun this command."
  exit 1
fi

rm -rf "$EAIP_CACHE"
mkdir -p "$EAIP_CACHE"
unzip -q "$EAIP_ZIP" -d "$EAIP_CACHE"

BACKEND_PYTHON="$BACKEND_DIR/.venv/bin/python"
if [[ ! -x "$BACKEND_PYTHON" ]]; then
  echo "Create the backend environment first; backend/.venv/bin/python is missing."
  exit 1
fi
"$BACKEND_PYTHON" -m venv "$BACKEND_DIR/.venv-eaip"
"$BACKEND_DIR/.venv-eaip/bin/python" -m pip install --upgrade pip
"$BACKEND_DIR/.venv-eaip/bin/python" -m pip install -r "$BACKEND_DIR/requirements-eaip.txt"

echo "EAIP-DARV runtime is ready."
echo "Run the app with ./run_app.sh llama or ./run_app.sh mistral."
