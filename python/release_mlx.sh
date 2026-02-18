#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUJOCO_REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
UPLOAD_TESTPYPI=0
SMOKE_TESTPYPI=0

usage() {
  cat <<'EOF'
Usage:
  bash release_mlx.sh [--upload-testpypi] [--smoke-testpypi]

Options:
  --upload-testpypi   Upload dist/* to TestPyPI via ~/.pypirc (repo: testpypi)
  --smoke-testpypi    After upload, install from TestPyPI in a clean venv and smoke test
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upload-testpypi)
      UPLOAD_TESTPYPI=1
      shift
      ;;
    --smoke-testpypi)
      SMOKE_TESTPYPI=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

activate_conda_env() {
  if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    if conda activate mlx >/dev/null 2>&1; then
      echo "[env] activated conda env: mlx"
      return 0
    fi
  fi
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    if conda activate mj_env >/dev/null 2>&1; then
      echo "[env] activated conda env: mj_env"
      return 0
    fi
  fi
  echo "[env] failed to activate 'mlx' or 'mj_env'" >&2
  exit 1
}

get_version() {
  awk -F'"' '/^version = / {print $2; exit}' pyproject.toml
}

activate_conda_env
cd "$ROOT_DIR"

VERSION="$(get_version)"
UPSTREAM_VERSION="${VERSION%%.post*}"
echo "[info] package version: ${VERSION}"
echo "[info] upstream version: ${UPSTREAM_VERSION}"

python -m pip install --upgrade build twine

pip install --force-reinstall --no-deps "mujoco==${UPSTREAM_VERSION}"
MUJOCO_SITE="$(python - <<'PY'
import inspect
import os
import mujoco

print(os.path.dirname(inspect.getfile(mujoco)))
PY
)"

export MUJOCO_PATH="${MUJOCO_SITE}"
export MUJOCO_PLUGIN_PATH="${MUJOCO_SITE}/plugin"
export MUJOCO_CMAKE_ARGS="-DCMAKE_MODULE_PATH:PATH=${MUJOCO_REPO_ROOT}/cmake;${ROOT_DIR}/mujoco/cmake"

# Ensure generated binding files exist before sdist/wheel build.
PYTHONPATH="${ROOT_DIR}/mujoco" \
python "${ROOT_DIR}/mujoco/codegen/generate_enum_traits.py" \
  > "${ROOT_DIR}/mujoco/enum_traits.h"

PYTHONPATH="${ROOT_DIR}/mujoco" \
python "${ROOT_DIR}/mujoco/codegen/generate_function_traits.py" \
  > "${ROOT_DIR}/mujoco/function_traits.h"

PYTHONPATH="${ROOT_DIR}/mujoco" \
python "${ROOT_DIR}/mujoco/codegen/generate_spec_bindings.py" \
  > "${ROOT_DIR}/mujoco/specs.cc.inc"

rm -rf dist build ./*.egg-info
# Build sdist and wheel separately:
# - sdist does not compile
# - wheel uses source tree (not sdist tempdir), so local simulate fallback works
python -m build --sdist
python -m build --wheel --no-isolation
python -m twine check dist/*

if [[ "$UPLOAD_TESTPYPI" -eq 1 ]]; then
  python -m twine upload -r testpypi dist/*
fi

if [[ "$SMOKE_TESTPYPI" -eq 1 ]]; then
  TMP_VENV_DIR="$(mktemp -d)"
  python -m venv "${TMP_VENV_DIR}/venv"
  # shellcheck disable=SC1091
  source "${TMP_VENV_DIR}/venv/bin/activate"
  pip install --upgrade pip
  pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    "mujoco-mlx==${VERSION}"
  python -c "import mujoco; from mujoco import mlx_step; print(mujoco.__version__, mlx_step.__name__)"
  deactivate
  rm -rf "${TMP_VENV_DIR}"
fi

echo "[done] build pipeline completed"
