#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  echo "Create the virtualenv first, for example: python3 -m venv .venv" >&2
  exit 1
fi

SITE_PACKAGES="$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_ROOT="$SITE_PACKAGES/nvidia"

if [[ -d "$NVIDIA_ROOT" ]]; then
  mapfile -t NVIDIA_LIB_DIRS < <(find "$NVIDIA_ROOT" -type d -name lib | sort)
  if (( ${#NVIDIA_LIB_DIRS[@]} > 0 )); then
    CUDA_LIBRARY_PATH="$(IFS=:; echo "${NVIDIA_LIB_DIRS[*]}")"
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
      export LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH:$LD_LIBRARY_PATH"
    else
      export LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH"
    fi
  fi
fi

exec "$PYTHON_BIN" "$ROOT_DIR/main.py" "$@"
