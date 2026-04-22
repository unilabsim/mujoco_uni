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
MANYLINUX_2_28=0
CONTAINER_RUNTIME=""
CONTAINER_ARCH="auto"
CONTROL_PYTHON_VERSION="${MUJOCO_RELEASE_CONTROL_PYTHON:-3.10}"
UV_RELEASE_DIR="${ROOT_DIR}/.release-uv"
CONTROL_PYTHON=""
TMP_WORK_ROOT=""
USER_MUJOCO_CMAKE_ARGS="${MUJOCO_CMAKE_ARGS:-}"
ALL_PYTHON_SPECS=(3.10 3.11 3.12 3.13)

usage() {
  cat <<'EOF'
Usage:
  bash release.sh [--upload-testpypi] [--upload-pypi] [--smoke-testpypi] [--build-all-python] [--upload-only] [--skip-existing] [--wheel-only] [--sdist-only] [--manylinux-2-28] [--container-runtime docker|podman] [--container-arch x86_64|aarch64]

Options:
  --upload-testpypi   Upload dist/* to TestPyPI via ~/.pypirc (repo: testpypi)
  --upload-pypi       Upload dist/* to official PyPI via ~/.pypirc (repo: pypi)
  --smoke-testpypi    After upload, install from TestPyPI in a clean uv venv and smoke test
  --build-all-python  Build wheels for Python 3.10, 3.11, 3.12, 3.13 via uv in parallel
  --upload-only       Skip build, only upload existing dist/* (use when build succeeded but upload failed)
  --skip-existing     Pass --skip-existing to twine upload (ignore files already on index)
  --wheel-only        Build wheel only (skip sdist)
  --sdist-only        Build sdist only (skip wheel)
  --manylinux-2-28    Build Linux wheel in a manylinux_2_28 container
  --container-runtime Container runtime used with --manylinux-2-28 (docker or podman)
  --container-arch    Container architecture used with --manylinux-2-28 (default: auto)

Environment:
  MUJOCO_RELEASE_CONTROL_PYTHON
                     Python version uv uses for release tooling (default: 3.10)

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
      echo "[warn] --no-auto-conda is deprecated and ignored; uv is always used" >&2
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

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "[error] uv is required. Install it from https://docs.astral.sh/uv/ and rerun." >&2
    exit 1
  fi
}

is_python_path() {
  case "${1:-}" in
    /*|./*|../*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

setup_uv_env() {
  local python_spec="$1"
  local env_dir="$2"
  shift 2

  if ! is_python_path "$python_spec"; then
    uv python install "$python_spec"
  fi

  rm -rf "$env_dir"
  mkdir -p "$(dirname "$env_dir")"
  uv venv --python "$python_spec" "$env_dir"
  uv pip install --python "${env_dir}/bin/python" --upgrade "$@"
}

setup_release_tools() {
  local include_mujoco="$1"
  local packages=(build setuptools twine wheel absl-py)

  if [[ "$(uname -s)" == "Linux" ]]; then
    packages+=(auditwheel)
  fi
  if [[ "$include_mujoco" -eq 1 ]]; then
    packages+=("mujoco==${UPSTREAM_VERSION}")
  fi

  echo "[env] preparing uv release env: Python ${CONTROL_PYTHON_VERSION}"
  setup_uv_env "$CONTROL_PYTHON_VERSION" "${UV_RELEASE_DIR}/control" "${packages[@]}"
  CONTROL_PYTHON="${UV_RELEASE_DIR}/control/bin/python"
}

get_current_python_spec() {
  if command -v python >/dev/null 2>&1; then
    python -c 'import sys; print(sys.executable)'
  else
    printf '%s\n' "$CONTROL_PYTHON"
  fi
}

python_label_for_spec() {
  local python_spec="$1"
  if [[ -x "$python_spec" ]]; then
    "$python_spec" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
  else
    printf '%s\n' "$python_spec"
  fi
}

safe_label() {
  printf '%s' "$1" | tr '/. ' '___' | tr -cd '[:alnum:]_-'
}

get_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    printf '%s\n' 1
  fi
}

recommended_cmake_parallel_level() {
  local job_count="$1"
  local cpu_count
  local level

  cpu_count="$(get_cpu_count)"
  if [[ -z "$cpu_count" || "$cpu_count" -lt 1 ]]; then
    cpu_count=1
  fi
  level=$((cpu_count / job_count))
  if [[ "$level" -lt 1 ]]; then
    level=1
  fi
  printf '%s\n' "$level"
}

copy_release_workspace() {
  local target_repo_root="$1"

  mkdir -p "$target_repo_root"
  tar -C "$MUJOCO_REPO_ROOT" \
    --exclude='./.git' \
    --exclude='./python/dist' \
    --exclude='./python/build' \
    --exclude='./python/.release-uv' \
    --exclude='./python/*.egg-info' \
    -cf - . | tar -C "$target_repo_root" -xf -
}

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
    current_pyver="$(python_label_for_spec "$(get_current_python_spec)")"
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
      USER_MUJOCO_CMAKE_ARGS="${USER_MUJOCO_CMAKE_ARGS:-} -DCMAKE_OSX_ARCHITECTURES=arm64"
    elif [[ "$host_arch" == "x86_64" ]]; then
      USER_MUJOCO_CMAKE_ARGS="${USER_MUJOCO_CMAKE_ARGS:-} -DCMAKE_OSX_ARCHITECTURES=x86_64"
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
  echo "      MUJOCO_CMAKE_ARGS=${USER_MUJOCO_CMAKE_ARGS:-<unset>}"
}

get_version() {
  awk -F'"' '/^version = / {print $2; exit}' pyproject.toml
}

get_upstream_version() {
  printf '%s\n' "$1" | cut -d. -f1-3
}

get_installed_mujoco_site() {
  local python_bin="${1:-python}"
  "$python_bin" -c 'from importlib.metadata import distribution; import pathlib; print(pathlib.Path(distribution("mujoco").locate_file("mujoco")).resolve())'
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
    manylinux_tag="$("$CONTROL_PYTHON" -m auditwheel show "$wheel_file" 2>/dev/null | sed -n 's/.*constrains the platform tag to "\(manylinux_[^"]*\)".*/\1/p' | head -n1)"
    if [[ -z "$manylinux_tag" ]]; then
      continue
    fi
    echo "[info] retag wheel for index compatibility: $(basename "$wheel_file") -> ${manylinux_tag}"
    "$CONTROL_PYTHON" -m wheel tags --platform-tag "$manylinux_tag" --remove "$wheel_file" >/dev/null
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
  local cmd=("$CONTROL_PYTHON" -m twine upload -r "$repo_name")

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
    manylinux_tag="$("$CONTROL_PYTHON" -m auditwheel show "$wheel_file" 2>/dev/null | sed -n 's/.*constrains the platform tag to "\(manylinux_[^"]*\)".*/\1/p' | head -n1)"
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

cleanup_release_workspace() {
  cleanup_bundled_assets
  if [[ -n "${TMP_WORK_ROOT:-}" ]]; then
    rm -rf "$TMP_WORK_ROOT"
  fi
}

prepare_bundled_assets() {
  local mujoco_site="$1"
  local bundle_dir="${ROOT_DIR}/mujoco"
  local bundle_file

  cleanup_bundled_assets
  mkdir -p "${bundle_dir}/include/mujoco" "${bundle_dir}/plugin" "${bundle_dir}/cmake"
  cp -f "${MUJOCO_REPO_ROOT}"/cmake/*.cmake "${bundle_dir}/cmake/" 2>/dev/null || true
  cp -rf "${MUJOCO_REPO_ROOT}"/simulate "${bundle_dir}/" 2>/dev/null || true
  for bundle_file in "${mujoco_site}"/libmujoco*.dylib "${mujoco_site}"/libmujoco*.so*; do
    [[ -e "$bundle_file" ]] && cp -f "$bundle_file" "${bundle_dir}/"
  done
  if [[ -d "${mujoco_site}/include/mujoco" ]]; then
    cp -f "${mujoco_site}/include/mujoco/"*.h "${bundle_dir}/include/mujoco/" 2>/dev/null || true
  fi
  if [[ -d "${mujoco_site}/plugin" ]]; then
    cp -f "${mujoco_site}/plugin/"* "${bundle_dir}/plugin/" 2>/dev/null || true
  fi
}

generate_binding_sources() {
  PYTHONPATH="${ROOT_DIR}/mujoco" \
  "$CONTROL_PYTHON" "${ROOT_DIR}/mujoco/codegen/generate_enum_traits.py" \
    > "${ROOT_DIR}/mujoco/enum_traits.h"

  PYTHONPATH="${ROOT_DIR}/mujoco" \
  "$CONTROL_PYTHON" "${ROOT_DIR}/mujoco/codegen/generate_function_traits.py" \
    > "${ROOT_DIR}/mujoco/function_traits.h"

  PYTHONPATH="${ROOT_DIR}/mujoco" \
  "$CONTROL_PYTHON" "${ROOT_DIR}/mujoco/codegen/generate_spec_bindings.py" \
    > "${ROOT_DIR}/mujoco/specs.cc.inc"
}

build_sdist_artifact() {
  "$CONTROL_PYTHON" -m build --sdist
}

build_wheel_for_python_spec() {
  local python_spec="$1"
  local work_root="$2"
  local wheels_dir="$3"
  local cmake_parallel_level="$4"
  local python_label
  local safe_python_label
  local env_dir
  local worker_repo_root
  local worker_python_root
  local env_python
  local worker_mujoco_site
  local worker_cmake_args
  local packages=(build setuptools wheel absl-py "mujoco==${UPSTREAM_VERSION}")

  python_label="$(python_label_for_spec "$python_spec")"
  safe_python_label="$(safe_label "$python_label")"
  env_dir="${UV_RELEASE_DIR}/py${safe_python_label}"
  worker_repo_root="${work_root}/repo-py${safe_python_label}"
  worker_python_root="${worker_repo_root}/python"

  echo "[info] Python ${python_label}: preparing uv build env"
  setup_uv_env "$python_spec" "$env_dir" "${packages[@]}"
  env_python="${env_dir}/bin/python"
  worker_mujoco_site="$(get_installed_mujoco_site "$env_python")"

  rm -rf "$worker_repo_root"
  copy_release_workspace "$worker_repo_root"
  rm -rf "${worker_python_root}/dist" "${worker_python_root}/build" "${worker_python_root}"/*.egg-info

  worker_cmake_args="${USER_MUJOCO_CMAKE_ARGS:-} -DCMAKE_MODULE_PATH:PATH=${worker_repo_root}/cmake;${worker_python_root}/mujoco/cmake"

  (
    cd "$worker_python_root"
    export MUJOCO_PATH="$worker_mujoco_site"
    export MUJOCO_PLUGIN_PATH="${worker_mujoco_site}/plugin"
    export MUJOCO_CMAKE_ARGS="$worker_cmake_args"
    if [[ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
      export CMAKE_BUILD_PARALLEL_LEVEL="$cmake_parallel_level"
    fi
    "$env_python" -m build --wheel --no-isolation
  )

  cp -f "${worker_python_root}"/dist/*.whl "$wheels_dir/"
}

build_host_wheels_parallel() {
  local python_specs=("$@")
  local work_root="${TMP_WORK_ROOT}/host-wheels"
  local wheels_dir="${work_root}/wheels"
  local logs_dir="${work_root}/logs"
  local cmake_parallel_level
  local python_spec
  local python_label
  local safe_python_label
  local log_file
  local wait_index
  local failed=0
  local -a pids=()
  local -a labels=()

  mkdir -p "$wheels_dir" "$logs_dir" "$ROOT_DIR/dist"
  cmake_parallel_level="$(recommended_cmake_parallel_level "${#python_specs[@]}")"
  echo "[info] building Python wheels in parallel: ${python_specs[*]}"
  if [[ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
    echo "[info] per-wheel CMake parallel level: ${cmake_parallel_level}"
  fi

  for python_spec in "${python_specs[@]}"; do
    python_label="$(python_label_for_spec "$python_spec")"
    safe_python_label="$(safe_label "$python_label")"
    log_file="${logs_dir}/py${safe_python_label}.log"
    (
      if build_wheel_for_python_spec "$python_spec" "$work_root" "$wheels_dir" "$cmake_parallel_level" >"$log_file" 2>&1; then
        echo "[done] Python ${python_label} wheel built"
      else
        echo "[error] Python ${python_label} wheel failed" >&2
        cat "$log_file" >&2
        exit 1
      fi
    ) &
    pids+=("$!")
    labels+=("$python_label")
  done

  for wait_index in "${!pids[@]}"; do
    if ! wait "${pids[$wait_index]}"; then
      echo "[error] Python ${labels[$wait_index]} build did not complete" >&2
      failed=1
    fi
  done

  if [[ "$failed" -eq 1 ]]; then
    exit 1
  fi

  shopt -s nullglob
  local built_wheels=("${wheels_dir}"/*.whl)
  shopt -u nullglob
  if [[ "${#built_wheels[@]}" -eq 0 ]]; then
    echo "[error] no wheels were built" >&2
    exit 1
  fi

  cp -f "${built_wheels[@]}" "${ROOT_DIR}/dist/"
}

run_smoke_testpypi() {
  local tmp_venv
  tmp_venv="$(mktemp -d)"
  uv venv --python "$CONTROL_PYTHON_VERSION" "${tmp_venv}/venv"
  uv pip install --python "${tmp_venv}/venv/bin/python" \
    -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    "mujoco-uni==${VERSION}"
  "${tmp_venv}/venv/bin/python" -c "import mujoco; print(mujoco.__version__)"
  rm -rf "$tmp_venv"
}

cd "$ROOT_DIR"
require_uv
TMP_WORK_ROOT="$(mktemp -d)"
trap cleanup_release_workspace EXIT
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
  setup_release_tools 0
  retag_linux_wheels_for_upload
  validate_linux_wheels_platform_floor
  "$CONTROL_PYTHON" -m twine check dist/*
  if [[ "$UPLOAD_TESTPYPI" -eq 1 ]]; then
    upload_dist_with_twine testpypi
  fi
  if [[ "$UPLOAD_PYPI" -eq 1 ]]; then
    upload_dist_with_twine pypi
  fi
  if [[ "$SMOKE_TESTPYPI" -eq 1 ]]; then
    run_smoke_testpypi
  fi
  echo "[done] upload completed"
  exit 0
fi

setup_release_tools 1
MUJOCO_SITE="$(get_installed_mujoco_site "$CONTROL_PYTHON")"

export MUJOCO_PATH="${MUJOCO_SITE}"
export MUJOCO_PLUGIN_PATH="${MUJOCO_SITE}/plugin"
export MUJOCO_CMAKE_ARGS="${USER_MUJOCO_CMAKE_ARGS:-} -DCMAKE_MODULE_PATH:PATH=${MUJOCO_REPO_ROOT}/cmake;${ROOT_DIR}/mujoco/cmake"

prepare_bundled_assets "$MUJOCO_SITE"
generate_binding_sources

if [[ "$MANYLINUX_2_28" -eq 1 ]]; then
  prepare_manylinux_2_28_workspace
fi

rm -rf dist build ./*.egg-info
mkdir -p dist

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
    if [[ "$BUILD_ALL_PYTHON" -eq 1 ]]; then
      build_host_wheels_parallel "${ALL_PYTHON_SPECS[@]}"
    else
      build_host_wheels_parallel "$(get_current_python_spec)"
    fi
  fi
fi

retag_linux_wheels_for_upload
validate_linux_wheels_platform_floor
"$CONTROL_PYTHON" -m twine check dist/*

if [[ "$UPLOAD_TESTPYPI" -eq 1 ]]; then
  upload_dist_with_twine testpypi
fi

if [[ "$UPLOAD_PYPI" -eq 1 ]]; then
  upload_dist_with_twine pypi
fi

if [[ "$SMOKE_TESTPYPI" -eq 1 ]]; then
  run_smoke_testpypi
fi

echo "[done] release pipeline completed"
