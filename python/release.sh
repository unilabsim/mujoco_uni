#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUJOCO_REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
UPLOAD_TESTPYPI=0
UPLOAD_PYPI=0
SMOKE_TESTPYPI=0
BUILD_ALL_PYTHON=0
UPLOAD_ONLY=0
SKIP_EXISTING=0
WHEEL_ONLY=0
SDIST_ONLY=0
NO_AUTO_CONDA=0

usage() {
  cat <<'EOF'
Usage:
  bash release.sh [--upload-testpypi] [--upload-pypi] [--smoke-testpypi] [--build-all-python] [--upload-only] [--skip-existing] [--wheel-only] [--sdist-only] [--no-auto-conda]

Options:
  --upload-testpypi   Upload dist/* to TestPyPI via ~/.pypirc (repo: testpypi)
  --upload-pypi       Upload dist/* to official PyPI via ~/.pypirc (repo: pypi)
  --smoke-testpypi    After upload, install from TestPyPI in a clean venv and smoke test
  --build-all-python  Build wheels for Python 3.10, 3.11, 3.12, 3.13 via conda
  --upload-only       Skip build, only upload existing dist/* (use when build succeeded but upload failed)
  --skip-existing     Pass --skip-existing to twine upload (ignore files already on index)
  --wheel-only        Build wheel only (skip sdist)
  --sdist-only        Build sdist only (skip wheel)
  --no-auto-conda     Use the current Python environment instead of auto-activating conda
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
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --wheel-only)
      WHEEL_ONLY=1
      shift
      ;;
    --sdist-only)
      SDIST_ONLY=1
      shift
      ;;
    --no-auto-conda)
      NO_AUTO_CONDA=1
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

if [[ "$WHEEL_ONLY" -eq 1 && "$SDIST_ONLY" -eq 1 ]]; then
  echo "[error] --wheel-only and --sdist-only are mutually exclusive" >&2
  exit 1
fi

activate_conda_env() {
  local conda_init_script=""
  local fallback_env="mj_wheel_py310"
  local env_name=""

  [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]] && conda_init_script="$HOME/anaconda3/etc/profile.d/conda.sh"
  [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] && conda_init_script="$HOME/miniconda3/etc/profile.d/conda.sh"

  if [[ -z "$conda_init_script" ]]; then
    echo "[env] conda init script not found (checked anaconda3/miniconda3)" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$conda_init_script"

  for env_name in mujoco_uni mj_env; do
    if conda activate "$env_name" >/dev/null 2>&1; then
      echo "[env] activated conda env: ${env_name}"
      return 0
    fi
  done

  echo "[env] no preferred conda env found (checked: mujoco_uni, mj_env); fallback to: ${fallback_env}"
  if conda activate "$fallback_env" >/dev/null 2>&1; then
    echo "[env] activated conda env: ${fallback_env}"
    return 0
  fi

  echo "[env] creating fallback conda env: ${fallback_env} (python=3.10)"
  conda create -n "$fallback_env" python=3.10 -y
  conda activate "$fallback_env"
  echo "[env] activated newly created conda env: ${fallback_env}"
}

get_version() {
  awk -F'"' '/^version = / {print $2; exit}' pyproject.toml
}

get_upstream_version() {
  printf '%s\n' "$1" | cut -d. -f1-3
}

retag_linux_wheels_for_upload() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    return 0
  fi

  shopt -s nullglob
  local wheel_file
  local retagged=0
  for wheel_file in "$ROOT_DIR"/dist/*-linux_*.whl; do
    local manylinux_tag
    manylinux_tag="$(python -m auditwheel show "$wheel_file" 2>/dev/null | sed -n 's/.*constrains the platform tag to "\(manylinux_[^"]*\)".*/\1/p' | head -n1)"
    if [[ -z "$manylinux_tag" ]]; then
      continue
    fi
    echo "[info] retag wheel for index compatibility: $(basename "$wheel_file") -> ${manylinux_tag}"
    python -m wheel tags --platform-tag "$manylinux_tag" --remove "$wheel_file" >/dev/null
    retagged=1
  done
  shopt -u nullglob

  if [[ "$retagged" -eq 1 ]]; then
    echo "[info] refreshed dist/ after Linux wheel retagging"
    ls -1 "$ROOT_DIR"/dist
  fi
}

cleanup_bundled_assets() {
  local bundle_dir="${ROOT_DIR}/mujoco"
  rm -rf "${bundle_dir}/include" "${bundle_dir}/plugin" "${bundle_dir}/cmake" "${bundle_dir}/simulate"
  rm -f "${bundle_dir}"/libmujoco*.dylib "${bundle_dir}"/libmujoco*.so*
}

if [[ "$NO_AUTO_CONDA" -eq 0 ]]; then
  activate_conda_env
fi

cd "$ROOT_DIR"
trap cleanup_bundled_assets EXIT

VERSION="$(get_version)"
UPSTREAM_VERSION="$(get_upstream_version "$VERSION")"
TWINE_UPLOAD_ARGS=()
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  TWINE_UPLOAD_ARGS+=(--skip-existing)
fi
echo "[info] package version: ${VERSION}"
echo "[info] upstream version: ${UPSTREAM_VERSION}"

if [[ "$UPLOAD_ONLY" -eq 1 ]]; then
  echo "[info] --upload-only: skipping build, uploading existing dist/*"
  if [[ ! -d "$ROOT_DIR/dist" ]] || [[ -z "$(ls -A "$ROOT_DIR/dist" 2>/dev/null)" ]]; then
    echo "[error] dist/ is empty, run full build first" >&2
    exit 1
  fi
  python -m pip install -q twine wheel 2>/dev/null || true
  if [[ "$(uname -s)" == "Linux" ]]; then
    python -m pip install -q auditwheel 2>/dev/null || true
  fi
  retag_linux_wheels_for_upload
  python -m twine check dist/*
  if [[ "$UPLOAD_TESTPYPI" -eq 1 ]]; then
    python -m twine upload "${TWINE_UPLOAD_ARGS[@]}" -r testpypi dist/*
  fi
  if [[ "$UPLOAD_PYPI" -eq 1 ]]; then
    python -m twine upload "${TWINE_UPLOAD_ARGS[@]}" -r pypi dist/*
  fi
  if [[ "$SMOKE_TESTPYPI" -eq 1 ]]; then
    TMP_VENV="$(mktemp -d)"
    python -m venv "${TMP_VENV}/venv"
    # shellcheck disable=SC1091
    source "${TMP_VENV}/venv/bin/activate"
    pip install -q -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "mujoco-uni==${VERSION}"
    python -c "import mujoco; print(mujoco.__version__)"
    deactivate 2>/dev/null
    rm -rf "${TMP_VENV}"
  fi
  echo "[done] upload completed"
  exit 0
fi

python -m pip install --upgrade build twine wheel
if [[ "$(uname -s)" == "Linux" ]]; then
  python -m pip install --upgrade auditwheel
fi

pip install --force-reinstall --no-deps "mujoco==${UPSTREAM_VERSION}"
MUJOCO_SITE="$(pip show mujoco | awk -F': ' '/^Location:/ {print $2"/mujoco"; exit}')"

export MUJOCO_PATH="${MUJOCO_SITE}"
export MUJOCO_PLUGIN_PATH="${MUJOCO_SITE}/plugin"
export MUJOCO_CMAKE_ARGS="-DCMAKE_MODULE_PATH:PATH=${MUJOCO_REPO_ROOT}/cmake;${ROOT_DIR}/mujoco/cmake"

BUNDLE_DIR="${ROOT_DIR}/mujoco"
cleanup_bundled_assets
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
if [[ "$WHEEL_ONLY" -eq 0 ]]; then
  python -m build --sdist
fi

if [[ "$SDIST_ONLY" -eq 0 ]]; then
  python -m build --wheel --no-isolation
fi

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
    for pyver in 3.10 3.11 3.12 3.13; do
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

retag_linux_wheels_for_upload

if [[ "$UPLOAD_TESTPYPI" -eq 1 ]]; then
  python -m twine upload "${TWINE_UPLOAD_ARGS[@]}" -r testpypi dist/*
fi

if [[ "$UPLOAD_PYPI" -eq 1 ]]; then
  python -m twine upload "${TWINE_UPLOAD_ARGS[@]}" -r pypi dist/*
fi

if [[ "$SMOKE_TESTPYPI" -eq 1 ]]; then
  TMP_VENV="$(mktemp -d)"
  python -m venv "${TMP_VENV}/venv"
  # shellcheck disable=SC1091
  source "${TMP_VENV}/venv/bin/activate"
  pip install --upgrade pip
  pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "mujoco-uni==${VERSION}"
  python -c "import mujoco; print(mujoco.__version__)"
  deactivate 2>/dev/null
  rm -rf "${TMP_VENV}"
fi

echo "[done] release pipeline completed"
