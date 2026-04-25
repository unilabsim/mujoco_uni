#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd -P)"
RUNTIME=""
ARCH="auto"
CONTAINER_NAME=""
FIX_PERMISSIONS_ONLY=0
PYTHON_VERSIONS=()

usage() {
  cat <<'EOF'
Usage:
  bash build_manylinux_2_28.sh [--runtime docker|podman] [--arch x86_64|aarch64] [--container-name NAME] [--fix-permissions-only] [--python 3.10]...

Options:
  --runtime        Container runtime to use. Defaults to docker, then podman.
  --arch           Container architecture. Defaults to the host architecture.
  --container-name Reuse or create this long-lived build container.
                   Defaults to: mujoco-manylinux-228-dev
  --fix-permissions-only
                   Only fix ownership of build outputs in the mounted workspace.
  --python         CPython version to build. Repeat to build multiple wheels.
                   Defaults to: 3.10 3.11 3.12 3.13

Examples:
  bash python/build_manylinux_2_28.sh
  bash python/build_manylinux_2_28.sh --python 3.10 --python 3.12

This reuses a fixed manylinux_2_28 container, creates uv-managed build envs,
and builds the requested Python wheels in parallel.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime)
      RUNTIME="${2:-}"
      shift 2
      ;;
    --arch)
      ARCH="${2:-}"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="${2:-}"
      shift 2
      ;;
    --fix-permissions-only)
      FIX_PERMISSIONS_ONLY=1
      shift
      ;;
    --python)
      PYTHON_VERSIONS+=("${2:-}")
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

if [[ -z "$RUNTIME" ]]; then
  if command -v docker >/dev/null 2>&1; then
    RUNTIME="docker"
  elif command -v podman >/dev/null 2>&1; then
    RUNTIME="podman"
  else
    echo "[error] docker or podman is required" >&2
    exit 1
  fi
fi

if [[ "$ARCH" == "auto" ]]; then
  case "$(uname -m)" in
    x86_64|amd64)
      ARCH="x86_64"
      ;;
    aarch64|arm64)
      ARCH="aarch64"
      ;;
    *)
      echo "[error] unsupported host architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
fi

if [[ -z "$CONTAINER_NAME" ]]; then
  if [[ "$ARCH" == "x86_64" ]]; then
    CONTAINER_NAME="mujoco-manylinux-228-dev"
  else
    CONTAINER_NAME="mujoco-manylinux-228-dev-${ARCH}"
  fi
fi

if [[ "${#PYTHON_VERSIONS[@]}" -eq 0 ]]; then
  PYTHON_VERSIONS=(3.10 3.11 3.12 3.13)
fi

map_python_to_abi() {
  case "$1" in
    3.10) printf '%s\n' "cp310-cp310" ;;
    3.11) printf '%s\n' "cp311-cp311" ;;
    3.12) printf '%s\n' "cp312-cp312" ;;
    3.13) printf '%s\n' "cp313-cp313" ;;
    *)
      echo "[error] unsupported Python version: $1" >&2
      exit 1
      ;;
  esac
}

ensure_container_exists() {
  local image="$1"
  local container_state=""
  local mounted_repo=""
  local configured_workdir=""
  local configured_image=""

  container_state="$("${RUNTIME}" ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Status}}' | head -n1)"
  if [[ -z "$container_state" ]]; then
    echo "[info] creating build container: ${CONTAINER_NAME}"
    "${RUNTIME}" run -d --name "${CONTAINER_NAME}" \
      -v "${REPO_ROOT}:/work" \
      -w /work \
      "${image}" \
      /bin/bash -lc 'trap : TERM INT; while sleep 3600; do :; done' >/dev/null
    return 0
  fi

  mounted_repo="$("${RUNTIME}" inspect "${CONTAINER_NAME}" \
    --format '{{range .Mounts}}{{if eq .Destination "/work"}}{{.Source}}{{end}}{{end}}')"
  configured_workdir="$("${RUNTIME}" inspect "${CONTAINER_NAME}" \
    --format '{{.Config.WorkingDir}}')"
  configured_image="$("${RUNTIME}" inspect "${CONTAINER_NAME}" \
    --format '{{.Config.Image}}')"

  if [[ -z "$mounted_repo" || "${mounted_repo}" != "${REPO_ROOT}" || \
        "${configured_workdir}" != "/work" || "${configured_image}" != "${image}" ]]; then
    echo "[info] recreating container ${CONTAINER_NAME} for current workspace"
    if [[ "$container_state" == Up* ]]; then
      "${RUNTIME}" stop "${CONTAINER_NAME}" >/dev/null
    fi
    "${RUNTIME}" rm -f "${CONTAINER_NAME}" >/dev/null
    "${RUNTIME}" run -d --name "${CONTAINER_NAME}" \
      -v "${REPO_ROOT}:/work" \
      -w /work \
      "${image}" \
      /bin/bash -lc 'trap : TERM INT; while sleep 3600; do :; done' >/dev/null
    return 0
  fi

  if [[ "$container_state" != Up* ]]; then
    echo "[info] starting existing build container: ${CONTAINER_NAME}"
    "${RUNTIME}" start "${CONTAINER_NAME}" >/dev/null
  else
    echo "[info] reusing running build container: ${CONTAINER_NAME}"
  fi
}

install_container_deps_once() {
  "${RUNTIME}" exec "${CONTAINER_NAME}" /bin/bash -lc '
    set -euo pipefail
    stamp=/tmp/mujoco-manylinux-228.deps.libcxx.v2.ready
    if [[ -f "${stamp}" ]]; then
      exit 0
    fi
    if command -v dnf >/dev/null 2>&1; then
      dnf -y install \
        clang \
        llvm \
        ninja-build \
        mesa-libGL-devel \
        wayland-devel \
        wayland-protocols-devel \
        libXcursor-devel \
        libXinerama-devel \
        libXrandr-devel \
        libXi-devel \
        libxkbcommon-devel >/dev/null
    fi
    touch "${stamp}"
  '
}

install_libcxx_once() {
  "${RUNTIME}" exec "${CONTAINER_NAME}" /bin/bash -lc '
    set -euo pipefail
    version="${MUJOCO_LIBCXX_LLVM_VERSION:-18.1.8}"
    prefix=/opt/mujoco-uni-libcxx
    stamp="${prefix}/.pic-ready-${version}"
    if [[ -f "${stamp}" ]]; then
      exit 0
    fi

    work=/tmp/mujoco-uni-libcxx-build
    src="${work}/llvm-project-${version}.src"
    archive="${work}/llvm-project-${version}.src.tar.xz"
    rm -rf "${work}"
    mkdir -p "${work}" "${prefix}"

    curl -fsSL \
      "https://github.com/llvm/llvm-project/releases/download/llvmorg-${version}/llvm-project-${version}.src.tar.xz" \
      -o "${archive}"
    tar -C "${work}" -xf "${archive}"

    cmake -S "${src}/runtimes" -B "${work}/build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_INSTALL_PREFIX="${prefix}" \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
      -DLIBCXX_ENABLE_SHARED=OFF \
      -DLIBCXX_ENABLE_STATIC=ON \
      -DLIBCXXABI_ENABLE_SHARED=OFF \
      -DLIBCXXABI_ENABLE_STATIC=ON \
      -DLIBCXXABI_USE_LLVM_UNWINDER=ON \
      -DLIBUNWIND_ENABLE_SHARED=OFF \
      -DLIBUNWIND_ENABLE_STATIC=ON \
      -DLIBCXX_INCLUDE_BENCHMARKS=OFF \
      -DLIBCXX_INCLUDE_TESTS=OFF \
      -DLIBCXXABI_INCLUDE_TESTS=OFF \
      -DLIBUNWIND_INCLUDE_TESTS=OFF
    cmake --build "${work}/build" --target install --parallel "$(nproc)"
    touch "${stamp}"
  '
}

prepare_workspace_permissions() {
  local host_uid="$1"
  local host_gid="$2"
  "${RUNTIME}" exec "${CONTAINER_NAME}" /bin/bash -lc "
    set -euo pipefail
    cd /work/python
    shopt -s nullglob
    targets=(build dist .release-uv ./*.egg-info)
    for target in \"\${targets[@]}\"; do
      [[ -e \"\${target}\" ]] || continue
      chown -R ${host_uid}:${host_gid} \"\${target}\" 2>/dev/null || true
    done
  "
}

IMAGE="quay.io/pypa/manylinux_2_28_${ARCH}"
ABI_TAGS=()
for pyver in "${PYTHON_VERSIONS[@]}"; do
  ABI_TAGS+=("$(map_python_to_abi "$pyver")")
done

echo "[info] runtime: ${RUNTIME}"
echo "[info] image: ${IMAGE}"
echo "[info] container: ${CONTAINER_NAME}"
echo "[info] python ABIs: ${ABI_TAGS[*]}"

ensure_container_exists "${IMAGE}"
install_container_deps_once
install_libcxx_once
prepare_workspace_permissions "$(id -u)" "$(id -g)"

if [[ "${FIX_PERMISSIONS_ONLY}" -eq 1 ]]; then
  echo "[done] workspace permissions prepared"
  exit 0
fi

abi_args=()
for abi in "${ABI_TAGS[@]}"; do
  abi_args+=("$abi")
done

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
CONTAINER_HOME="/tmp/mujoco-uni-home-${HOST_UID}"
CONTAINER_CACHE_HOME="${CONTAINER_HOME}/.cache"
CONTAINER_PIP_CACHE_DIR="${CONTAINER_CACHE_HOME}/pip"
CONTAINER_UV_CACHE_DIR="${CONTAINER_CACHE_HOME}/uv"

"${RUNTIME}" exec \
  -u "${HOST_UID}:${HOST_GID}" \
  -e HOME="${CONTAINER_HOME}" \
  -e XDG_CACHE_HOME="${CONTAINER_CACHE_HOME}" \
  -e PIP_CACHE_DIR="${CONTAINER_PIP_CACHE_DIR}" \
  -e UV_CACHE_DIR="${CONTAINER_UV_CACHE_DIR}" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -e MUJOCO_CMAKE_ARGS="${MUJOCO_CMAKE_ARGS:-}" \
  "${CONTAINER_NAME}" \
  /bin/bash -lc "$(cat <<'EOF'
set -euo pipefail
cd /work/python

export HOME="${HOME:-/tmp/mujoco-uni-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}"

build_root=/tmp/mujoco-uni-manylinux-build
preserve_dir="${build_root}/preserve-dist"
wheels_dir="${build_root}/wheels"
logs_dir="${build_root}/logs"
venvs_dir="${build_root}/venvs"
workspaces_dir="${build_root}/workspaces"
bootstrap_dir="${build_root}/uv-bootstrap"
rm -rf "${preserve_dir}" "${wheels_dir}" "${logs_dir}" "${venvs_dir}" "${workspaces_dir}"
mkdir -p "${preserve_dir}" "${wheels_dir}" "${logs_dir}" "${venvs_dir}" "${workspaces_dir}"

if [[ -d dist ]]; then
  shopt -s nullglob
  for artifact in dist/*; do
    case "${artifact}" in
      *.whl)
        continue
        ;;
    esac
    cp -a "${artifact}" "${preserve_dir}/"
  done
  shopt -u nullglob
fi

get_upstream_version() {
  awk -F'"' '/^version = / {print $2; exit}' /work/python/pyproject.toml | cut -d. -f1-3
}

get_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  else
    getconf _NPROCESSORS_ONLN 2>/dev/null || printf '%s\n' 1
  fi
}

recommended_cmake_parallel_level() {
  local job_count="$1"
  local cpu_count
  local level

  cpu_count="$(get_cpu_count)"
  if [[ -z "${cpu_count}" || "${cpu_count}" -lt 1 ]]; then
    cpu_count=1
  fi
  level=$((cpu_count / job_count))
  if [[ "${level}" -lt 1 ]]; then
    level=1
  fi
  printf '%s\n' "${level}"
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

ensure_uv_available() {
  local bootstrap_python="$1"
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "[info] bootstrapping uv inside container"
  rm -rf "${bootstrap_dir}"
  "${bootstrap_python}" -m venv "${bootstrap_dir}"
  "${bootstrap_dir}/bin/python" -m pip install --quiet --upgrade pip uv
  export PATH="${bootstrap_dir}/bin:${PATH}"
}

setup_uv_env() {
  local python_spec="$1"
  local env_dir="$2"
  shift 2

  if ! is_python_path "${python_spec}"; then
    uv python install "${python_spec}"
  fi

  rm -rf "${env_dir}"
  uv venv --python "${python_spec}" "${env_dir}"
  uv pip install --python "${env_dir}/bin/python" --upgrade "$@"
}

get_installed_mujoco_site() {
  local python_bin="$1"
  "${python_bin}" -c 'from importlib.metadata import distribution; import pathlib; print(pathlib.Path(distribution("mujoco").locate_file("mujoco")).resolve())'
}

copy_release_workspace() {
  local target_repo_root="$1"
  mkdir -p "${target_repo_root}"
  tar -C /work \
    --exclude='./.git' \
    --exclude='./python/dist' \
    --exclude='./python/build' \
    --exclude='./python/.release-uv' \
    --exclude='./python/*.egg-info' \
    -cf - . | tar -C "${target_repo_root}" -xf -
}

cleanup_bundled_assets() {
  local python_root="$1"
  local bundle_dir="${python_root}/mujoco"
  rm -rf "${bundle_dir}/include" "${bundle_dir}/plugin" "${bundle_dir}/cmake" "${bundle_dir}/simulate"
  rm -f "${bundle_dir}"/libmujoco*.dylib "${bundle_dir}"/libmujoco*.so*
}

prepare_bundled_assets() {
  local repo_root="$1"
  local python_root="$2"
  local mujoco_site="$3"
  local bundle_dir="${python_root}/mujoco"
  local bundle_file

  cleanup_bundled_assets "${python_root}"
  mkdir -p "${bundle_dir}/include/mujoco" "${bundle_dir}/plugin" "${bundle_dir}/cmake"
  cp -f "${repo_root}"/cmake/*.cmake "${bundle_dir}/cmake/" 2>/dev/null || true
  cp -rf "${repo_root}"/simulate "${bundle_dir}/" 2>/dev/null || true
  for bundle_file in "${mujoco_site}"/libmujoco*.dylib "${mujoco_site}"/libmujoco*.so*; do
    [[ -e "${bundle_file}" ]] && cp -f "${bundle_file}" "${bundle_dir}/"
  done
  if [[ -d "${mujoco_site}/include/mujoco" ]]; then
    cp -f "${mujoco_site}/include/mujoco/"*.h "${bundle_dir}/include/mujoco/" 2>/dev/null || true
  fi
  if [[ -d "${mujoco_site}/plugin" ]]; then
    cp -f "${mujoco_site}/plugin/"* "${bundle_dir}/plugin/" 2>/dev/null || true
  fi
}

generate_binding_sources() {
  local python_root="$1"
  local python_bin="$2"

  PYTHONPATH="${python_root}/mujoco" \
  "${python_bin}" "${python_root}/mujoco/codegen/generate_enum_traits.py" \
    > "${python_root}/mujoco/enum_traits.h"

  PYTHONPATH="${python_root}/mujoco" \
  "${python_bin}" "${python_root}/mujoco/codegen/generate_function_traits.py" \
    > "${python_root}/mujoco/function_traits.h"

  PYTHONPATH="${python_root}/mujoco" \
  "${python_bin}" "${python_root}/mujoco/codegen/generate_spec_bindings.py" \
    > "${python_root}/mujoco/specs.cc.inc"
}

run_batch_env_smoke() {
  local python_bin="$1"
  "${python_bin}" - <<'PY'
import mujoco
from mujoco.batch_env import SUPPORTED_FIELDS

assert "gravity" in SUPPORTED_FIELDS, SUPPORTED_FIELDS
spec = mujoco.MjSpec()
body = spec.worldbody.add_body()
body.name = "abi_ok"
assert body.name == "abi_ok", body.name
print("mujoco", mujoco.__version__)
print("batch_env fields", sorted(SUPPORTED_FIELDS))
PY
}

smoke_wheel_for_python() {
  local python_bin="$1"
  local wheel_path="$2"
  local smoke_env_dir="${build_root}/smoke-venvs/$(basename "${python_bin}")-$(basename "${wheel_path}")"

  rm -rf "${smoke_env_dir}"
  uv venv --python "${python_bin}" "${smoke_env_dir}"
  uv pip install --python "${smoke_env_dir}/bin/python" --no-deps "${wheel_path}"
  run_batch_env_smoke "${smoke_env_dir}/bin/python"
}

build_one_abi() {
  local abi="$1"
  local cmake_parallel_level="$2"
  local pybin="/opt/python/${abi}/bin/python"
  local env_dir="${venvs_dir}/${abi}"
  local worker_repo_root="${workspaces_dir}/${abi}"
  local worker_python_root="${worker_repo_root}/python"
  local env_python
  local mujoco_site
  local worker_cmake_args
  local ar_bin
  local ranlib_bin

  if [[ ! -x "${pybin}" ]]; then
    echo "[error] missing interpreter: ${pybin}" >&2
    exit 1
  fi

  setup_uv_env "${pybin}" "${env_dir}" build setuptools wheel absl-py "mujoco==${UPSTREAM_VERSION}"
  env_python="${env_dir}/bin/python"
  mujoco_site="$(get_installed_mujoco_site "${env_python}")"

  rm -rf "${worker_repo_root}"
  copy_release_workspace "${worker_repo_root}"
  rm -rf "${worker_python_root}/dist" "${worker_python_root}/build" "${worker_python_root}"/*.egg-info
  prepare_bundled_assets "${worker_repo_root}" "${worker_python_root}" "${mujoco_site}"
  generate_binding_sources "${worker_python_root}" "${env_python}"
  ar_bin="$(command -v llvm-ar || command -v ar)"
  ranlib_bin="$(command -v llvm-ranlib || command -v ranlib)"

  worker_cmake_args="${MUJOCO_CMAKE_ARGS:-} -DMUJOCO_PYTHON_USE_LIBCXX:BOOL=ON -DMUJOCO_PYTHON_LIBCXX_ROOT:PATH=/opt/mujoco-uni-libcxx -DCMAKE_C_COMPILER:STRING=clang -DCMAKE_CXX_COMPILER:STRING=clang++ -DCMAKE_AR:FILEPATH=${ar_bin} -DCMAKE_RANLIB:FILEPATH=${ranlib_bin} -DCMAKE_CXX_COMPILER_AR:FILEPATH=${ar_bin} -DCMAKE_CXX_COMPILER_RANLIB:FILEPATH=${ranlib_bin} -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=bfd -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=bfd -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=bfd -DCMAKE_MODULE_PATH:PATH=${worker_repo_root}/cmake;${worker_python_root}/mujoco/cmake"

  (
    cd "${worker_python_root}"
    export MUJOCO_PATH="${mujoco_site}"
    export MUJOCO_PLUGIN_PATH="${mujoco_site}/plugin"
    export MUJOCO_CMAKE_ARGS="${worker_cmake_args}"
    if [[ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
      export CMAKE_BUILD_PARALLEL_LEVEL="${cmake_parallel_level}"
    fi
    "${env_python}" -m build --wheel --no-isolation
  ) || return

  smoke_wheel_for_python "${pybin}" "${worker_python_root}"/dist/*.whl
  cp -f "${worker_python_root}"/dist/*.whl "${wheels_dir}/"
}

UPSTREAM_VERSION="$(get_upstream_version)"
cmake_parallel_level="$(recommended_cmake_parallel_level "$#")"
first_pybin="/opt/python/$1/bin/python"
ensure_uv_available "${first_pybin}"

echo "[info] upstream version: ${UPSTREAM_VERSION}"
echo "[info] per-wheel CMake parallel level: ${cmake_parallel_level}"
echo "[info] building ABIs in parallel: $*"

declare -a pids=()
declare -a labels=()

for abi in "$@"; do
  log_file="${logs_dir}/${abi}.log"
  (
    if build_one_abi "${abi}" "${cmake_parallel_level}" >"${log_file}" 2>&1; then
      echo "[done] ${abi} wheel built"
    else
      echo "[error] ${abi} wheel failed" >&2
      cat "${log_file}" >&2
      exit 1
    fi
  ) &
  pids+=("$!")
  labels+=("${abi}")
done

build_failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[error] ${labels[$index]} build did not complete" >&2
    build_failed=1
  fi
done

if [[ "${build_failed}" -ne 0 ]]; then
  exit 1
fi

rm -rf dist
mkdir -p dist
cp -a "${preserve_dir}/". dist/ 2>/dev/null || true
mv "${wheels_dir}/"*.whl dist/

echo "[done] built wheels:"
ls -1 dist/*.whl
EOF
)" _ "${abi_args[@]}"
