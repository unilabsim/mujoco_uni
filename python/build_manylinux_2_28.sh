#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
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

This reuses a fixed manylinux_2_28 container so Linux wheels can be built,
retagged, and later uploaded together with host-built sdists.
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
    stamp=/tmp/mujoco-manylinux-228.deps.ready
    if [[ -f "${stamp}" ]]; then
      exit 0
    fi
    if command -v dnf >/dev/null 2>&1; then
      dnf -y install \
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

prepare_workspace_permissions() {
  local host_uid="$1"
  local host_gid="$2"
  "${RUNTIME}" exec "${CONTAINER_NAME}" /bin/bash -lc "
    set -euo pipefail
    cd /work/python
    shopt -s nullglob
    targets=(build dist dist_all_manylinux_2_27 ./*.egg-info)
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
prepare_workspace_permissions "$(id -u)" "$(id -g)"

if [[ "${FIX_PERMISSIONS_ONLY}" -eq 1 ]]; then
  echo "[done] workspace permissions prepared"
  exit 0
fi

abi_args=()
for abi in "${ABI_TAGS[@]}"; do
  abi_args+=("$abi")
done

"${RUNTIME}" exec \
  -u "$(id -u):$(id -g)" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -e MUJOCO_CMAKE_ARGS="${MUJOCO_CMAKE_ARGS:-}" \
  "${CONTAINER_NAME}" \
  /bin/bash -lc "$(cat <<'EOF'
set -euo pipefail
cd /work/python

mkdir -p /tmp/mujoco-uni-manylinux-build
preserve_dir=/tmp/mujoco-uni-manylinux-build/preserve-dist
wheels_dir=/tmp/mujoco-uni-manylinux-build/wheels
rm -rf "${preserve_dir}" "${wheels_dir}"
mkdir -p "${preserve_dir}" "${wheels_dir}"

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

for abi in "$@"; do
  pybin="/opt/python/${abi}/bin/python"
  if [[ ! -x "${pybin}" ]]; then
    echo "[error] missing interpreter: ${pybin}" >&2
    exit 1
  fi

  venv="/tmp/venv-${abi}"
  rm -rf "${venv}"
  "${pybin}" -m venv "${venv}"
  # shellcheck disable=SC1090
  source "${venv}/bin/activate"

  python -m pip install --upgrade pip
  export MUJOCO_CMAKE_ARGS="${MUJOCO_CMAKE_ARGS:-} -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=bfd -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=bfd -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=bfd"
  bash release.sh --wheel-only --no-auto-conda
  mv dist/*.whl "${wheels_dir}/"

  deactivate
done

rm -rf dist
mkdir -p dist
cp -a "${preserve_dir}/". dist/ 2>/dev/null || true
mv "${wheels_dir}/"*.whl dist/

echo "[done] built wheels:"
ls -1 dist/*.whl
EOF
)" _ "${abi_args[@]}"
