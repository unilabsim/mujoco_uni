#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUJOCO_REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
UPLOAD_TESTPYPI=0
UPLOAD_PYPI=0
SMOKE_TESTPYPI=0
BUILD_ALL_PYTHON=0
UPLOAD_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash release.sh [--upload-testpypi] [--upload-pypi] [--smoke-testpypi] [--build-all-python] [--upload-only]

Options:
  --upload-testpypi   Upload dist/* to TestPyPI via ~/.pypirc (repo: testpypi)
  --upload-pypi       Upload dist/* to official PyPI via ~/.pypirc (repo: pypi)
  --smoke-testpypi    After upload, install from TestPyPI in a clean venv and smoke test
  --build-all-python  Build wheels for Python 3.9, 3.10, 3.11, 3.12 via conda (avoids local compile for users)
  --upload-only       Skip build, only upload existing dist/* (use when build succeeded but upload failed)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upload-testpypi)
      UPLOAD_TESTPYPI=1
      shift
      ;;
    --upload-pypi)
      UPLOAD_PYPI=1
      shift
      ;;
    --smoke-testpypi)
      SMOKE_TESTPYPI=1
      shift
      ;;
    --build-all-python)
      BUILD_ALL_PYTHON=1
      shift
      ;;
    --upload-only)
      UPLOAD_ONLY=1
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

if [[ "$UPLOAD_ONLY" -eq 1 ]]; then
  echo "[info] --upload-only: skipping build, uploading existing dist/*"
  if [[ ! -d "$ROOT_DIR/dist" ]] || [[ -z "$(ls -A "$ROOT_DIR/dist" 2>/dev/null)" ]]; then
    echo "[error] dist/ is empty, run full build first" >&2
    exit 1
  fi
  python -m pip install -q twine 2>/dev/null || true
  python -m twine check dist/*
  if [[ "$UPLOAD_TESTPYPI" -eq 1 ]]; then
    python -m twine upload -r testpypi dist/*
  fi
  if [[ "$UPLOAD_PYPI" -eq 1 ]]; then
    python -m twine upload -r pypi dist/*
  fi
  if [[ "$SMOKE_TESTPYPI" -eq 1 ]]; then
    TMP_VENV="$(mktemp -d)"
    python -m venv "${TMP_VENV}/venv"
    # shellcheck disable=SC1091
    source "${TMP_VENV}/venv/bin/activate"
    pip install -q -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "mujoco-uni==${VERSION}"
    python -c "import mujoco; print(mujoco.__version__)"
    deactivate 2>/dev/null; rm -rf "${TMP_VENV}"
  fi
  echo "[done] upload completed"
  exit 0
fi

python -m pip install --upgrade build twine

pip install --force-reinstall --no-deps "mujoco==${UPSTREAM_VERSION}"
# Get installed mujoco path (avoid picking up local project copy)
MUJOCO_SITE="$(pip show mujoco | awk -F': ' '/^Location:/ {print $2"/mujoco"; exit}')"

export MUJOCO_PATH="${MUJOCO_SITE}"
export MUJOCO_PLUGIN_PATH="${MUJOCO_SITE}/plugin"
export MUJOCO_CMAKE_ARGS="-DCMAKE_MODULE_PATH:PATH=${MUJOCO_REPO_ROOT}/cmake;${ROOT_DIR}/mujoco/cmake"

# Bundle mujoco lib/headers/cmake/simulate into package so sdist can be built from source (no MUJOCO_PATH)
BUNDLE_DIR="${ROOT_DIR}/mujoco"
mkdir -p "${BUNDLE_DIR}/include/mujoco" "${BUNDLE_DIR}/plugin" "${BUNDLE_DIR}/cmake"
cp -f "${MUJOCO_REPO_ROOT}"/cmake/*.cmake "${BUNDLE_DIR}/cmake/" 2>/dev/null || true
cp -rf "${MUJOCO_REPO_ROOT}"/simulate "${BUNDLE_DIR}/" 2>/dev/null || true
for f in "${MUJOCO_SITE}"/libmujoco*.dylib "${MUJOCO_SITE}"/libmujoco*.so*; do
  [[ -e "$f" ]] && cp -f "$f" "${BUNDLE_DIR}/"
done
if [[ -d "${MUJOCO_SITE}/include/mujoco" ]]; then
  cp -f "${MUJOCO_SITE}/include/mujoco/"*.h "${BUNDLE_DIR}/include/mujoco/" 2>/dev/null || true
fi
if [[ -d "${MUJOCO_SITE}/plugin" ]]; then
  cp -f "${MUJOCO_SITE}/plugin/"* "${BUNDLE_DIR}/plugin/" 2>/dev/null || true
fi

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
# - sdist includes bundled lib/headers for pip install from source (no MUJOCO_PATH)
# - wheel uses source tree (not sdist tempdir), so local simulate fallback works
python -m build --sdist
python -m build --wheel --no-isolation

# Build wheels for other Python versions: PATH binaries first, then conda if --build-all-python
for py in python3.9 python3.10 python3.11 python3.12; do
  if command -v "$py" &>/dev/null && [[ "$("$py" -c 'import sys; print(sys.executable)' 2>/dev/null)" != "$(python -c 'import sys; print(sys.executable)' 2>/dev/null)" ]]; then
    echo "[info] building wheel for $("$py" --version 2>&1)"
    MUJOCO_PATH="${MUJOCO_SITE}" MUJOCO_PLUGIN_PATH="${MUJOCO_SITE}/plugin" \
      "$py" -m build --wheel --no-isolation 2>/dev/null || true
  fi
done

# Build wheels for Python 3.9, 3.10, 3.11, 3.12 via conda (so users get pre-built wheels, no compile)
if [[ "$BUILD_ALL_PYTHON" -eq 1 ]]; then
  conda_init_script=""
  [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]] && conda_init_script="$HOME/anaconda3/etc/profile.d/conda.sh"
  [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] && conda_init_script="$HOME/miniconda3/etc/profile.d/conda.sh"
  if [[ -z "$conda_init_script" ]]; then
    echo "[warn] conda not found, skip --build-all-python"
  else
    # shellcheck disable=SC1090
    source "$conda_init_script"
    CURRENT_PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    for pyver in 3.9 3.10 3.11 3.12; do
      if [[ "$pyver" == "$CURRENT_PY_VER" ]]; then
        echo "[info] Python $pyver wheel already built (current env)"
        continue
      fi
      env_name="mj_wheel_py${pyver//./}"
      echo "[info] building wheel for Python $pyver (conda env: $env_name)"
      conda create -n "$env_name" python="$pyver" -y 2>/dev/null || conda install -n "$env_name" python="$pyver" -y 2>/dev/null || true
      conda run -n "$env_name" --no-capture-output bash -c "
        pip install -q 'mujoco==${UPSTREAM_VERSION}' build
        MUJOCO_SITE=\$(pip show mujoco | awk -F': ' '/^Location:/ {print \$2\"/mujoco\"; exit}')
        export MUJOCO_PATH=\$MUJOCO_SITE
        export MUJOCO_PLUGIN_PATH=\$MUJOCO_SITE/plugin
        cd '${ROOT_DIR}' && python -m build --wheel --no-isolation
      " 2>/dev/null || echo "[warn] failed to build wheel for Python $pyver"
    done
  fi
fi

python -m twine check dist/*

if [[ "$UPLOAD_TESTPYPI" -eq 1 ]]; then
  python -m twine upload -r testpypi dist/*
fi
if [[ "$UPLOAD_PYPI" -eq 1 ]]; then
  python -m twine upload -r pypi dist/*
fi

if [[ "$SMOKE_TESTPYPI" -eq 1 ]]; then
  TMP_VENV_DIR="$(mktemp -d)"
  python -m venv "${TMP_VENV_DIR}/venv"
  # shellcheck disable=SC1091
  source "${TMP_VENV_DIR}/venv/bin/activate"
  pip install --upgrade pip
  pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    "mujoco-uni==${VERSION}"
  python -c "import mujoco; print(mujoco.__version__)"
  deactivate
  rm -rf "${TMP_VENV_DIR}"
fi

echo "[done] build pipeline completed"
