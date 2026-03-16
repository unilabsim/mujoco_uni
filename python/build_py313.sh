#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VERSION=$(awk -F'"' '/^version = / {print $2; exit}' pyproject.toml)
UPSTREAM_VERSION="${VERSION%%.post*}"

echo "[info] Building Python 3.13 wheel"
echo "[info] Package version: ${VERSION}"

pip install -q build twine wheel auditwheel
pip install --force-reinstall --no-deps "mujoco==${UPSTREAM_VERSION}"

MUJOCO_SITE=$(pip show mujoco | awk -F': ' '/^Location:/ {print $2"/mujoco"; exit}')
export MUJOCO_PATH="${MUJOCO_SITE}"
export MUJOCO_PLUGIN_PATH="${MUJOCO_SITE}/plugin"
export MUJOCO_CMAKE_ARGS="-DCMAKE_MODULE_PATH:PATH=$(pwd)/../cmake;$(pwd)/mujoco/cmake"

python -m build --wheel --no-isolation

# Retag wheel with manylinux
MANYLINUX_TAG=$(python -m auditwheel show dist/*cp313*linux*.whl 2>/dev/null | sed -n 's/.*constrains the platform tag to "\(manylinux_[^"]*\)".*/\1/p' | head -n1)
if [[ -n "$MANYLINUX_TAG" ]]; then
  echo "[info] Retagging wheel with ${MANYLINUX_TAG}"
  python -m wheel tags --platform-tag "$MANYLINUX_TAG" --remove dist/*cp313*linux*.whl
fi

echo "[info] Wheel built successfully"
ls -lh dist/*cp313*.whl
