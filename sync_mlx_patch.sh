#!/usr/bin/env bash
set -euo pipefail

# One-click patch queue migration:
# - export patches from upstream/vFROM .. mlx/vFROM
# - create/reset mlx/vTO from upstream/vTO
# - apply patches with git am
#
# Example:
#   ./sync_mlx_patch.sh --from 3.5.0 --to 3.6.0

usage() {
  cat <<'EOF'
Usage:
  ./sync_mlx_patch.sh --from <old_upstream_ver> --to <new_upstream_ver> [--keep-branch]

Required:
  --from    Old upstream version, e.g. 3.5.0
  --to      New upstream version, e.g. 3.6.0

Optional:
  --keep-branch   Do not reset existing mlx/v<to>; fail if missing

Branch convention:
  upstream/vX.Y.Z
  mlx/vX.Y.Z

Behavior:
  1) Export patches from upstream/vFROM..mlx/vFROM
  2) Create patches/vFROM/*.patch
  3) Create/reset mlx/vTO from upstream/vTO (unless --keep-branch)
  4) Apply patches via git am
EOF
}

require_clean_tree() {
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[error] git working tree is not clean. Commit/stash first." >&2
    exit 1
  fi
}

require_branch() {
  local branch="$1"
  if ! git show-ref --verify --quiet "refs/heads/${branch}"; then
    echo "[error] missing local branch: ${branch}" >&2
    exit 1
  fi
}

FROM_VER=""
TO_VER=""
KEEP_BRANCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      FROM_VER="${2:-}"
      shift 2
      ;;
    --to)
      TO_VER="${2:-}"
      shift 2
      ;;
    --keep-branch)
      KEEP_BRANCH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${FROM_VER}" || -z "${TO_VER}" ]]; then
  usage
  exit 1
fi

FROM_UP="upstream/v${FROM_VER}"
FROM_MLX="mlx/v${FROM_VER}"
TO_UP="upstream/v${TO_VER}"
TO_MLX="mlx/v${TO_VER}"
PATCH_DIR="patches/v${FROM_VER}"

require_clean_tree
require_branch "${FROM_UP}"
require_branch "${FROM_MLX}"
require_branch "${TO_UP}"

mkdir -p "${PATCH_DIR}"
rm -f "${PATCH_DIR}"/*.patch

echo "[step] export patch queue: ${FROM_UP}..${FROM_MLX}"
git format-patch --output-directory "${PATCH_DIR}" "${FROM_UP}..${FROM_MLX}"

if compgen -G "${PATCH_DIR}/*.patch" >/dev/null; then
  echo "[ok] exported patches to ${PATCH_DIR}"
else
  echo "[error] no patches exported from ${FROM_UP}..${FROM_MLX}" >&2
  exit 1
fi

if [[ "${KEEP_BRANCH}" -eq 1 ]]; then
  require_branch "${TO_MLX}"
  git checkout "${TO_MLX}"
else
  echo "[step] create/reset ${TO_MLX} from ${TO_UP}"
  git checkout -B "${TO_MLX}" "${TO_UP}"
fi

echo "[step] applying patch queue to ${TO_MLX}"
set +e
git am "${PATCH_DIR}"/*.patch
AM_EXIT=$?
set -e

if [[ "${AM_EXIT}" -ne 0 ]]; then
  cat <<EOF
[warn] git am failed (likely conflict).
Resolve conflicts, then run:
  git am --continue
Or abort with:
  git am --abort
EOF
  exit "${AM_EXIT}"
fi

cat <<EOF
[done] Patch queue migrated successfully.
Current branch: ${TO_MLX}

Next recommended steps:
  1) Update python/pyproject.toml version to ${TO_VER}.post1
  2) Run python/release_mlx.sh --upload-testpypi --smoke-testpypi
  3) Tag release: v${TO_VER}-mlx.1
EOF
