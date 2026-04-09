#!/usr/bin/env bash
set -euo pipefail

# Reapply Uni local patches from one MuJoCo release to the next.
#
# Examples:
#   ./sync_uni_patch.sh --to 3.7.0
#   ./sync_uni_patch.sh --from 3.5.0 --from-branch uni/v3.5.0 --to 3.6.0

usage() {
  cat <<'EOF'
Usage:
  ./sync_uni_patch.sh --to <new_upstream_tag> [--from <old_upstream_tag>] [--from-branch <uni_branch>] [--patch-dir <dir>] [--keep-branch] [--dry-run]

Required:
  --to           New official MuJoCo tag, e.g. 3.6.0

Optional:
  --from         Old official MuJoCo tag. Defaults to the version parsed from
                 the source Uni branch if it matches uni/v<version>.
  --from-branch  Source Uni branch. Defaults to the current branch.
  --patch-dir    Output directory for exported patches.
  --keep-branch  Do not reset existing uni/v<to>; fail if it is missing
  --dry-run      Export patches and print the diff summary, but do not switch
                 branches or apply patches

Convention:
  source local branch: uni/v<from>
  target local branch: uni/v<to>

Behavior:
  1) Compare <from> with the source Uni branch and print a summary
  2) Export patches from <from>..<source_uni_branch>
  3) Create patches/uni-v<from>-to-v<to>/*.patch
  4) Create/reset uni/v<to> from <to> (unless --keep-branch)
  5) Apply patches with git am -3

Recommended workflow:
  1) Checkout the current Uni branch, e.g. uni/v3.6.0
  2) Run ./sync_uni_patch.sh --to <next_official_tag>
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

require_not_in_progress() {
  local git_dir
  git_dir="$(git rev-parse --git-dir)"
  if [[ -d "${git_dir}/rebase-apply" || -d "${git_dir}/rebase-merge" ]]; then
    echo "[error] git am/rebase is already in progress. Resolve or abort it first." >&2
    exit 1
  fi
}

current_branch() {
  git symbolic-ref --quiet --short HEAD || true
}

infer_version_from_uni_branch() {
  local branch="$1"
  if [[ "${branch}" =~ ^uni/v(.+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

print_local_delta_summary() {
  local from_up="$1"
  local from_uni="$2"
  local to_up="$3"
  local to_uni="$4"

  echo "[info] source upstream tag: ${from_up}"
  echo "[info] source Uni branch:   ${from_uni}"
  echo "[info] target upstream tag: ${to_up}"
  echo "[info] target Uni branch:   ${to_uni}"
  echo

  echo "[info] local-only commits (${from_up}..${from_uni}):"
  git log --reverse --oneline "${from_up}..${from_uni}"
  echo

  echo "[info] diffstat (${from_up}..${from_uni}):"
  git diff --stat "${from_up}..${from_uni}"
  echo
}

FROM_VER=""
FROM_BRANCH=""
TO_VER=""
PATCH_DIR=""
KEEP_BRANCH=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      FROM_VER="${2:-}"
      shift 2
      ;;
    --from-branch)
      FROM_BRANCH="${2:-}"
      shift 2
      ;;
    --to)
      TO_VER="${2:-}"
      shift 2
      ;;
    --patch-dir)
      PATCH_DIR="${2:-}"
      shift 2
      ;;
    --keep-branch)
      KEEP_BRANCH=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ -z "${TO_VER}" ]]; then
  usage
  exit 1
fi

CURRENT_BRANCH="$(current_branch)"

if [[ -z "${FROM_BRANCH}" ]]; then
  FROM_BRANCH="${CURRENT_BRANCH}"
fi

if [[ -z "${FROM_BRANCH}" ]]; then
  echo "[error] failed to detect current branch. Pass --from-branch explicitly." >&2
  exit 1
fi

if [[ -z "${FROM_VER}" ]]; then
  if ! FROM_VER="$(infer_version_from_uni_branch "${FROM_BRANCH}")"; then
    echo "[error] failed to infer --from from branch '${FROM_BRANCH}'. Pass --from explicitly." >&2
    exit 1
  fi
fi

FROM_UP="${FROM_VER}"
TO_UP="${TO_VER}"
FROM_UNI="${FROM_BRANCH}"
TO_UNI="uni/v${TO_VER}"

if [[ -z "${PATCH_DIR}" ]]; then
  PATCH_DIR="patches/uni-v${FROM_VER}-to-v${TO_VER}"
fi

require_clean_tree
require_not_in_progress
require_ref "${FROM_UP}"
require_ref "${FROM_UNI}"
require_ref "${TO_UP}"

if [[ "${FROM_UNI}" == "${TO_UNI}" ]]; then
  echo "[error] source branch and target branch resolve to the same ref: ${FROM_UNI}" >&2
  exit 1
fi

if [[ "${FROM_UP}" == "${TO_UP}" ]]; then
  echo "[error] source upstream tag and target upstream tag are identical: ${FROM_UP}" >&2
  exit 1
fi

if [[ -z "$(git rev-list "${FROM_UP}..${FROM_UNI}")" ]]; then
  echo "[error] no local commits found in ${FROM_UP}..${FROM_UNI}" >&2
  exit 1
fi

print_local_delta_summary "${FROM_UP}" "${FROM_UNI}" "${TO_UP}" "${TO_UNI}"

mkdir -p "${PATCH_DIR}"
rm -f "${PATCH_DIR}"/*.patch

echo "[step] export patch queue: ${FROM_UP}..${FROM_UNI}"
git format-patch --full-index --binary --output-directory "${PATCH_DIR}" "${FROM_UP}..${FROM_UNI}"

if ! compgen -G "${PATCH_DIR}/*.patch" >/dev/null; then
  echo "[error] no patches exported from ${FROM_UP}..${FROM_UNI}" >&2
  exit 1
fi

echo "[ok] exported patches to ${PATCH_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  cat <<EOF
[done] dry-run completed.
No branch was switched and no patch was applied.

Next steps:
  1) Inspect exported patches in ${PATCH_DIR}
  2) Re-run without --dry-run to create ${TO_UNI}
EOF
  exit 0
fi

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
