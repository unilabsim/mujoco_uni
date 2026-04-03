#!/usr/bin/env bash
set -euo pipefail

# Reapply Uni local patches from one MuJoCo release to the next.
#
# Example:
#   ./sync_uni_patch.sh --from 3.5.0 --to 3.6.0

usage() {
  cat <<'EOF'
Usage:
  ./sync_uni_patch.sh --from <old_upstream_tag> --to <new_upstream_tag> [--keep-branch]

Required:
  --from    Old official MuJoCo tag, e.g. 3.5.0
  --to      New official MuJoCo tag, e.g. 3.6.0

Optional:
  --keep-branch   Do not reset existing uni/v<to>; fail if it is missing

Convention:
  old local branch: uni/v<from>
  new local branch: uni/v<to>

Behavior:
  1) Export patches from <from>..uni/v<from>
  2) Create patches/uni-v<from>-to-v<to>/*.patch
  3) Create/reset uni/v<to> from <to> (unless --keep-branch)
  4) Apply patches with git am -3
EOF
}

require_clean_tree() {
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[error] git working tree is not clean. Commit or stash first." >&2
    exit 1
  fi
}

require_ref() {
  local ref="$1"
  if ! git rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
    echo "[error] missing ref: ${ref}" >&2
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

FROM_UP="${FROM_VER}"
TO_UP="${TO_VER}"
FROM_UNI="uni/v${FROM_VER}"
TO_UNI="uni/v${TO_VER}"
PATCH_DIR="patches/uni-v${FROM_VER}-to-v${TO_VER}"

require_clean_tree
require_ref "${FROM_UP}"
require_ref "${FROM_UNI}"
require_ref "${TO_UP}"

mkdir -p "${PATCH_DIR}"
rm -f "${PATCH_DIR}"/*.patch

echo "[step] export patch queue: ${FROM_UP}..${FROM_UNI}"
git format-patch --full-index --binary --output-directory "${PATCH_DIR}" "${FROM_UP}..${FROM_UNI}"

if ! compgen -G "${PATCH_DIR}/*.patch" >/dev/null; then
  echo "[error] no patches exported from ${FROM_UP}..${FROM_UNI}" >&2
  exit 1
fi

echo "[ok] exported patches to ${PATCH_DIR}"

if [[ "${KEEP_BRANCH}" -eq 1 ]]; then
  require_ref "${TO_UNI}"
  git checkout "${TO_UNI}"
else
  echo "[step] create/reset ${TO_UNI} from ${TO_UP}"
  git checkout -B "${TO_UNI}" "${TO_UP}"
fi

echo "[step] apply patch queue to ${TO_UNI}"
set +e
git am -3 "${PATCH_DIR}"/*.patch
AM_EXIT=$?
set -e

if [[ "${AM_EXIT}" -ne 0 ]]; then
  cat <<EOF
[warn] git am failed, likely due to upstream drift.

Resolve conflicts, then continue with:
  git am --continue

Or abort with:
  git am --abort

Suggested inspection:
  git diff
  git status
  git range-diff ${FROM_UP}..${FROM_UNI} ${TO_UP}..${TO_UNI}
EOF
  exit "${AM_EXIT}"
fi

cat <<EOF
[done] Uni patch queue migrated successfully.
Current branch: ${TO_UNI}

Next recommended steps:
  1) Review package metadata in python/pyproject.toml
  2) Re-run build/test for the Python package
  3) Inspect drift with:
     git range-diff ${FROM_UP}..${FROM_UNI} ${TO_UP}..${TO_UNI}
EOF
