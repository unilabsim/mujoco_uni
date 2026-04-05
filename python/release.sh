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
MANYLINUX_2_28=0
CONTAINER_RUNTIME=""
CONTAINER_ARCH="auto"

usage() {
  cat <<'EOF'
Usage:
  bash release.sh [--upload-testpypi] [--upload-pypi] [--smoke-testpypi] [--build-all-python] [--upload-only] [--skip-existing] [--wheel-only] [--sdist-only] [--no-auto-conda] [--manylinux-2-28] [--container-runtime docker|podman] [--container-arch x86_64|aarch64]

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
  --manylinux-2-28    Build Linux wheel in a manylinux_2_28 container
  --container-runtime Container runtime used with --manylinux-2-28 (docker or podman)
  --container-arch    Container architecture used with --manylinux-2-28 (default: auto)

Linux note:
  Building on a recent host distro sets the wheel's real glibc floor to that
  host (for example Ubuntu 24.04 -> manylinux_2_39). If you need Ubuntu 22.04
  compatibility, pass --manylinux-2-28.
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
    --manylinux-2-28)
      MANYLINUX_2_28=1
      shift
      ;;
    --container-runtime)
      CONTAINER_RUNTIME="${2:-}"
      shift 2
      ;;
    --container-arch)
      CONTAINER_ARCH="${2:-}"
      shift 2
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

run_manylinux_2_28_wheel_build() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[error] --manylinux-2-28 is only supported on Linux hosts" >&2
    exit 1
  fi

  local helper_args=()
  local current_pyver=""

  if [[ -n "$CONTAINER_RUNTIME" ]]; then
    helper_args+=(--runtime "$CONTAINER_RUNTIME")
  fi
  if [[ "$CONTAINER_ARCH" != "auto" ]]; then
    helper_args+=(--arch "$CONTAINER_ARCH")
  fi
  if [[ "$BUILD_ALL_PYTHON" -eq 0 ]]; then
    current_pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    helper_args+=(--python "$current_pyver")
  fi

  bash "${ROOT_DIR}/build_manylinux_2_28.sh" "${helper_args[@]}"
}

prepare_manylinux_2_28_workspace() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[error] --manylinux-2-28 is only supported on Linux hosts" >&2
    exit 1
  fi

  local helper_args=(--fix-permissions-only)

  if [[ -n "$CONTAINER_RUNTIME" ]]; then
    helper_args+=(--runtime "$CONTAINER_RUNTIME")
  fi
  if [[ "$CONTAINER_ARCH" != "auto" ]]; then
    helper_args+=(--arch "$CONTAINER_ARCH")
  fi

  bash "${ROOT_DIR}/build_manylinux_2_28.sh" "${helper_args[@]}"
}

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

configure_macos_build_env() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    return 0
  fi

  local host_arch=""
  local deployment_tag=""
  host_arch="$(uname -m)"

  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
  deployment_tag="${MACOSX_DEPLOYMENT_TARGET}"

  if [[ -z "${ARCHFLAGS:-}" ]]; then
    if [[ "$host_arch" == "arm64" ]]; then
      export ARCHFLAGS="-arch arm64"
    elif [[ "$host_arch" == "x86_64" ]]; then
      export ARCHFLAGS="-arch x86_64"
    fi
  fi

  if [[ "${MUJOCO_CMAKE_ARGS:-}" != *"CMAKE_OSX_ARCHITECTURES"* ]]; then
    if [[ "$host_arch" == "arm64" ]]; then
      export MUJOCO_CMAKE_ARGS="${MUJOCO_CMAKE_ARGS:-} -DCMAKE_OSX_ARCHITECTURES=arm64"
    elif [[ "$host_arch" == "x86_64" ]]; then
      export MUJOCO_CMAKE_ARGS="${MUJOCO_CMAKE_ARGS:-} -DCMAKE_OSX_ARCHITECTURES=x86_64"
    fi
  fi

  if [[ -z "${_PYTHON_HOST_PLATFORM:-}" ]]; then
    if [[ "$host_arch" == "arm64" ]]; then
      export _PYTHON_HOST_PLATFORM="macosx-${deployment_tag}-arm64"
    elif [[ "$host_arch" == "x86_64" ]]; then
      export _PYTHON_HOST_PLATFORM="macosx-${deployment_tag}-x86_64"
    fi
  fi

  echo "[env] macOS build defaults:"
  echo "      MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}"
  echo "      _PYTHON_HOST_PLATFORM=${_PYTHON_HOST_PLATFORM:-<unset>}"
  echo "      ARCHFLAGS=${ARCHFLAGS:-<unset>}"
  echo "      MUJOCO_CMAKE_ARGS=${MUJOCO_CMAKE_ARGS:-<unset>}"
}

get_version() {
  awk -F'"' '/^version = / {print $2; exit}' pyproject.toml
}

get_upstream_version() {
  printf '%s\n' "$1" | cut -d. -f1-3
}

get_installed_mujoco_site() {
  python -c 'from importlib.metadata import distribution; import pathlib; print(pathlib.Path(distribution("mujoco").locate_file("mujoco")).resolve())'
}

manylinux_tag_is_compatible_with_2_28() {
  local tag="${1:-}"
  if [[ -z "$tag" ]]; then
    return 0
  fi

  case "$tag" in
    manylinux2014|manylinux2014_*|manylinux_2_17|manylinux_2_17_*|manylinux_2_24|manylinux_2_24_*|manylinux_2_27|manylinux_2_27_*|manylinux_2_28|manylinux_2_28_*)
      return 0
      ;;
    manylinux_*)
      if [[ "$tag" =~ ^manylinux_([0-9]+)_([0-9]+)(_|$) ]]; then
        local major="${BASH_REMATCH[1]}"
        local minor="${BASH_REMATCH[2]}"
        if (( major < 2 || (major == 2 && minor <= 28) )); then
          return 0
        fi
      fi
      return 1
      ;;
    *)
      return 0
      ;;
  esac
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

upload_dist_with_twine() {
  local repo_name="$1"
  local cmd=(python -m twine upload -r "$repo_name")

  if [[ "$SKIP_EXISTING" -eq 1 ]]; then
    cmd+=(--skip-existing)
  fi

  cmd+=(dist/*)
  "${cmd[@]}"
}

validate_linux_wheels_platform_floor() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    return 0
  fi

  shopt -s nullglob
  local wheel_file
  local incompatible=0
  for wheel_file in "$ROOT_DIR"/dist/*.whl; do
    local manylinux_tag
    manylinux_tag="$(python -m auditwheel show "$wheel_file" 2>/dev/null | sed -n 's/.*constrains the platform tag to "\(manylinux_[^"]*\)".*/\1/p' | head -n1)"
    if [[ -z "$manylinux_tag" ]]; then
      continue
    fi
    if ! manylinux_tag_is_compatible_with_2_28 "$manylinux_tag"; then
      echo "[error] incompatible Linux wheel detected: $(basename "$wheel_file") requires ${manylinux_tag}" >&2
      echo "[error] build Linux wheels inside python/build_manylinux_2_28.sh to target Ubuntu 22.04 / manylinux_2_28" >&2
      incompatible=1
    fi
  done
  shopt -u nullglob

  if [[ "$incompatible" -eq 1 ]]; then
    exit 1
  fi
}

cleanup_bundled_assets() {
  local bundle_dir="${ROOT_DIR}/mujoco"
  rm -rf "${bundle_dir}/include" "${bundle_dir}/plugin" "${bundle_dir}/cmake" "${bundle_dir}/simulate"
  rm -f "${bundle_dir}"/libmujoco*.dylib "${bundle_dir}"/libmujoco*.so*
}

build_sdist_artifact() {
  python -m build --sdist
}

if [[ "$NO_AUTO_CONDA" -eq 0 ]]; then
  activate_conda_env
fi

cd "$ROOT_DIR"
trap cleanup_bundled_assets EXIT
configure_macos_build_env

VERSION="$(get_version)"
UPSTREAM_VERSION="$(get_upstream_version "$VERSION")"
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
  validate_linux_wheels_platform_floor
  python -m twine check dist/*
  if [[ "$UPLOAD_TESTPYPI" -eq 1 ]]; then
    upload_dist_with_twine testpypi
  fi
  if [[ "$UPLOAD_PYPI" -eq 1 ]]; then
    upload_dist_with_twine pypi
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

python -m pip install --upgrade build setuptools twine wheel absl-py
if [[ "$(uname -s)" == "Linux" ]]; then
  python -m pip install --upgrade auditwheel
fi

pip install --force-reinstall --no-deps "mujoco==${UPSTREAM_VERSION}"
MUJOCO_SITE="$(get_installed_mujoco_site)"

export MUJOCO_PATH="${MUJOCO_SITE}"
export MUJOCO_PLUGIN_PATH="${MUJOCO_SITE}/plugin"
export MUJOCO_CMAKE_ARGS="${MUJOCO_CMAKE_ARGS:-} -DCMAKE_MODULE_PATH:PATH=${MUJOCO_REPO_ROOT}/cmake;${ROOT_DIR}/mujoco/cmake"

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

if [[ "$MANYLINUX_2_28" -eq 1 ]]; then
  prepare_manylinux_2_28_workspace
fi

rm -rf dist build ./*.egg-info

if [[ "$MANYLINUX_2_28" -eq 1 ]]; then
  if [[ "$SDIST_ONLY" -eq 0 ]]; then
    run_manylinux_2_28_wheel_build
  fi
  if [[ "$WHEEL_ONLY" -eq 0 ]]; then
    build_sdist_artifact
  fi
else
  if [[ "$WHEEL_ONLY" -eq 0 ]]; then
    build_sdist_artifact
  fi
  if [[ "$SDIST_ONLY" -eq 0 ]]; then
    python -m build --wheel --no-isolation
  fi
fi

if [[ "$BUILD_ALL_PYTHON" -eq 1 && "$MANYLINUX_2_28" -eq 0 ]]; then
  conda_init_script=""
  [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]] && conda_init_script="$HOME/anaconda3/etc/profile.d/conda.sh"
  [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] && conda_init_script="$HOME/miniconda3/etc/profile.d/conda.sh"
  if [[ -z "$conda_init_script" ]]; then
    echo "[warn] conda not found, skip --build-all-python"
  else
    # shellcheck disable=SC1090
    source "$conda_init_script"
    CONDA_BASE="$(conda info --base)"
    CURRENT_PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    for pyver in 3.10 3.11 3.12 3.13; do
      if [[ "$pyver" == "$CURRENT_PY_VER" ]]; then
        echo "[info] Python $pyver wheel already built (current env)"
        continue
      fi
      env_name="mj_wheel_py${pyver//./}"
      env_prefix="${CONDA_BASE}/envs/${env_name}"
      echo "[info] building wheel for Python $pyver (conda env: $env_name)"

      if [[ -d "$env_prefix" && ! -f "$env_prefix/conda-meta/history" ]]; then
        echo "[warn] skip Python $pyver: ${env_prefix} exists but is not a valid conda environment"
        echo "[warn] remove that directory or repair the environment, then rerun --build-all-python"
        continue
      fi

      conda create -n "$env_name" python="$pyver" -y || conda install -n "$env_name" python="$pyver" -y || true

      if ! conda run -n "$env_name" --no-capture-output python -V >/dev/null 2>&1; then
        echo "[warn] skip Python $pyver: conda env ${env_name} is unavailable after create/install"
        continue
      fi

      conda run -n "$env_name" --no-capture-output bash -c "
        export MACOSX_DEPLOYMENT_TARGET='${MACOSX_DEPLOYMENT_TARGET:-}'
        export _PYTHON_HOST_PLATFORM='${_PYTHON_HOST_PLATFORM:-}'
        export ARCHFLAGS='${ARCHFLAGS:-}'
        export MUJOCO_CMAKE_ARGS='${MUJOCO_CMAKE_ARGS:-}'
        pip install -q 'mujoco==${UPSTREAM_VERSION}' build setuptools absl-py
        MUJOCO_SITE=\$(python -c 'from importlib.metadata import distribution; import pathlib; print(pathlib.Path(distribution(\"mujoco\").locate_file(\"mujoco\")).resolve())')
        export MUJOCO_PATH=\$MUJOCO_SITE
        export MUJOCO_PLUGIN_PATH=\$MUJOCO_SITE/plugin
        cd '${ROOT_DIR}' && python -m build --wheel --no-isolation
      " || echo "[warn] failed to build wheel for Python $pyver"
    done
  fi
fi

retag_linux_wheels_for_upload
validate_linux_wheels_platform_floor
python -m twine check dist/*

if [[ "$UPLOAD_TESTPYPI" -eq 1 ]]; then
  upload_dist_with_twine testpypi
fi

if [[ "$UPLOAD_PYPI" -eq 1 ]]; then
  upload_dist_with_twine pypi
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
