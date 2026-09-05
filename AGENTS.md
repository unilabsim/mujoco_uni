# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

This repository is **MuJoCoUni** (PyPI package `mujoco-uni-runtime`, version `0.4.0`), a
standalone batched executor layer for **official MuJoCo**. It provides the
`BatchEnvPool` API consumed by the upstream **UniLab** project, without modifying
MuJoCo solver, contact, integrator, or source-tree internals. Licensed Apache-2.0.

Architecture (layered, bottom-up):

- **Official MuJoCo** (`mujoco>=3.5,<3.12`): compiler, `mjModel`/`mjData`, solver, C API.
- **Native C++ extension** (`mujoco_uni.compiled._batch_env`, pybind11, C++17): owns a
  per-environment `mjModel` pool (cloned with `mj_copyModel`), per-thread reusable
  `mjData` workers, and a local thread pool. Executes batched `mj_step` / `mj_forward` /
  fused sparse reset / site-Jacobian / hfield-height queries. No MPI, no OpenMP.
- **Python runtime** (`mujoco_uni.runtime`): input validation, compatibility checks,
  stable `BatchEnvPool` wrapper around the native module.
- **Public API** (`mujoco_uni.batch_env`): stable import path used by UniLab:
  `BatchEnvPool`, `SUPPORTED_FIELDS`, `batch_available`, `batch_import_error`.

## Repository Layout

```text
src/mujoco_uni/
  __init__.py                 # lazy top-level exports (metadata + public API)
  metadata.py                 # __version__, MuJoCo supported range/spec
  batch_env.py                # stable public import path (re-exports runtime API)
  runtime/
    __init__.py               # available_backends(), batch_diagnostics()
    batch.py                  # BatchEnvPool Python wrapper, validation, compat checks
  compiled/
    __init__.py               # preloads libmujoco, loads _batch_env*.so, diagnostics
  mujoco_runtime/
    api.py                    # MuJoCoUni-owned access point to official `mujoco`
    version_control.py        # MuJoCo solver-version discovery/selection/spawning
  native/
    batch_env.cc              # native pybind11 executor (~2200 lines)
    threadpool.h / .cc        # local worker thread pool
tests/                        # pytest suite (requires built native extension)
tools/
  version_matrix.py           # build/test the package across MuJoCo versions
  strict_ab_mj38.py           # strict A/B performance benchmark harness (mj 3.8)
                              # report: github.com/unilabsim/mujoco_uni/discussions/4
Makefile                      # common dev commands (sync/install/lint/test/matrix/...)
.github/workflows/release.yml # sdist build/test matrix + PyPI publish on v* tags
```

## Build and Test Commands

Environment management uses **uv**; dev dependencies are `pytest` and `ruff`
(`[dependency-groups] dev` in `pyproject.toml`).

The common commands are wrapped in the **Makefile** (style mirrors
`../UniLab/Makefile`):

```bash
make sync            # uv sync
make install         # editable install (compiles the native extension against
                     # the mujoco version currently in the active environment)
make mujoco MJ=3.10.0  # switch solver version + rebuild the native extension
make lint            # ruff check
make test            # pytest
make test-no-sync    # pytest with --no-sync (after `make mujoco MJ=...`)
make check           # lint + test
make matrix          # version-matrix checks across MuJoCo 3.5.0 / 3.6.0 / 3.7.0 /
                     # 3.8.0 / 3.8.1 / 3.9.0 / 3.10.0 / 3.11.0 (tools/version_matrix.py --pytest)
make test-unilab     # full UniLab task validation (sibling ../UniLab checkout)
make test-unilab-train  # UniLab training smoke per MuJoCo environment
make clean           # remove caches and build artifacts
```

Equivalent raw commands:

```bash
# Dev setup
uv sync
uv pip install --force-reinstall --no-deps --no-build-isolation -e .

# Validation
uv run ruff check .
uv run pytest -q

# After switching solver versions, keep the already-built native target:
uv run --no-sync pytest -q

# Version-matrix checks
uv run python tools/version_matrix.py --pytest
```

Critical build facts:

- The package is **source distribution only — no prebuilt wheels on purpose**. The
  native extension is compiled against the `mujoco` package present at build time and
  **refuses to load against any other MuJoCo version** (exact-version watchdog at
  import time). Always install with `--no-build-isolation` so the build sees the
  target environment's `mujoco`.
- Building from source requires a C++17 toolchain and Python dev headers (bundled in
  uv-managed Pythons).
- Editable installs generate a local artifact like
  `src/mujoco_uni/compiled/_batch_env.cpython-3XX-<platform>.so`, tied to the active
  Python + platform + MuJoCo patch version. **Rebuild after switching virtual
  environments, Python versions, or MuJoCo versions**:

  ```bash
  uv pip install "mujoco==3.10.0" pybind11 wheel
  uv pip install --force-reinstall --no-deps --no-build-isolation -e .
  ```

- `setup.py` contains a custom `build_ext`: it injects pybind11/numpy/mujoco include
  dirs, defines `MUJOCO_UNI_BUILD_MUJOCO_VERSION`, links directly against the
  `libmujoco*` / `mujoco.dll` shipped inside the `mujoco` wheel, sets rpath to
  `$ORIGIN/../../mujoco` (Linux) / `@loader_path/../../mujoco` (macOS), and generates
  an MSVC import library from referenced `mj*`/`mju*` symbols on Windows.

## MuJoCo Version Policy

- MuJoCoUni package version is independent of the MuJoCo solver version. Supported
  solver range: `mujoco>=3.5,<3.12` (`metadata.py`); default `3.11.0`.
- **One MuJoCo version per Python environment/process.** Version switching is
  process-level: the `MUJOCO_UNI_VERSION` env var requests a version
  (`3.8` = any compatible `3.8.x`, `3.8.0` = exact), and
  `mujoco_uni.mujoco_runtime.version_control` discovers existing versioned uv
  environments matching `.venv-mj*` under the working directory, verifies them, and
  spawns the target command there **before** `mujoco` is imported.
- Discovery preference order: `3.11 > 3.8 > 3.10 > 3.9 > 3.7 > 3.6 > 3.5`
  (`SUPPORTED_MUJOCO_MINOR_ORDER`). Missing/unusable requests fall back with a
  warning; no usable environment is a hard error.
- Import-time fail-fast checks (`runtime/batch.py`): loaded `mujoco` must be in range,
  must expose `mujoco.MjModel._address` and `mujoco.MjModel._from_model_ptr`, and must
  exactly match the native extension's recorded build version.

## Testing Instructions

- Test suite: `pytest`, files in `tests/` (no `conftest.py`):
  - `test_batch_env.py` — package metadata/version constants and basic pool behavior.
  - `test_batch_env_parity.py` — numerical parity of batched `step`/`forward`/`reset`,
    Jacobian, and hfield sampling against reference official-MuJoCo single-env runs
    (uses small inline MJCF models: pendulum, hfield terrain).
  - `test_cpu_affinity.py` — Linux-only `cpu_ids` worker pinning behavior.
  - `test_version_control.py` — version parsing/discovery/spawning logic.
- Tests require the native extension to be built **and matched to the installed
  `mujoco` version**; a stale `.so` fails import-time compatibility checks.
- Optional heavier validation (expects a sibling `../UniLab` checkout):
  `tools/version_matrix.py --unilab` / `--unilab-train`, plus the UniLab-side pytest
  and training-smoke commands documented in README.md.

## Code Style Guidelines

- **Ruff** (`pyproject.toml`): `line-length = 100`, `target-version = "py310"`,
  `src = ["src", "tests"]`, lint rules `E, F, I, N, W`; ignored: `E501`, `N806`,
  `E402`, `N801`, `N802`, `N803`. `E402` is ignored deliberately: modules must import
  in a specific order (compatibility checks before importing the native extension).
- Indentation is **not uniform**: `src/mujoco_uni/runtime/batch.py` uses **2-space**
  indentation; all other Python files use 4 spaces. Match the surrounding file.
- Python requires `>=3.10,<3.14`; code uses `from __future__ import annotations`
  and modern typing (`str | None`, `tuple[...]`).
- Public API stability matters: UniLab imports `from mujoco_uni.batch_env import
  BatchEnvPool, SUPPORTED_FIELDS`. Keep that path and its exported symbols stable.
- Docstrings/comments are in English (README_zh.md is a translation; keep it in sync
  when changing user-facing docs).

## Execution Model Notes (relevant when changing runtime/native code)

- `BatchEnvPool` construction: reads the official model pointer from Python, clones
  models via `mj_copyModel`, stores pool-owned models. Accepts one model or a
  compatible sequence of length `1` / `nbatch`.
- Hot path: native C++ calling official MuJoCo C API with one reusable `mjData` per
  worker thread, disjoint env chunks and output slots, one synchronization point per
  batch operation. `step` returns only the **final** state (optionally final
  sensordata), not trajectories.
- `cpu_ids` (Linux only) pins worker thread `i` to `cpu_ids[i]`; length must equal
  `nthread`; validated at construction; exposed read-only via `pool.cpu_ids` and
  `pool.worker_cpu_ids()`.
- Model views from `get_model` / `get_models` / `get_all_models` are **non-owning** and
  valid only while the pool is alive.
- Supported randomization fields (`SUPPORTED_FIELDS`): `body_mass`, `body_ipos`,
  `body_iquat`, `body_inertia`, `dof_armature`, `gravity`, `geom_friction`, `kp`, `kd`.
- Thread safety: built-in sensors via `mjData.sensordata` are supported; custom
  sensors, plugins, and global MuJoCo callbacks with mutable state are the caller's
  responsibility when `nthread > 1`.

## Security and Scope Boundaries

- Out of scope (do not implement here): MuJoCo source-code changes, solver/contact/
  integrator forks, distributed MPI factorization, OpenMP inside the executor, and
  UniLab task YAMLs/rollout/reward logic.
- The version-control layer spawns subprocesses (`run_in_env`) with environment
  variables set; treat command/version inputs as trusted local configuration, but keep
  validation strict (see `parse_version`, `verify_env`).
- No secrets are stored in the repo; PyPI publishing uses GitHub trusted publishing
  (OIDC), no API tokens.

## GitHub CLI (gh) Cheat Sheet

Remote: `github.com/unilabsim/mujoco_uni` (SSH). Authenticated via `gh`.

### Issues

```bash
gh issue list
gh issue view <number>
gh api repos/unilabsim/mujoco_uni/issues/<number> --jq '.title, .body'
```

### Discussions

Discussions are enabled on the repo. `gh` has no native discussion subcommand;
use GraphQL:

```bash
# List categories (needed once for the category id)
gh api graphql -f query='query { repository(owner:"unilabsim", name:"mujoco_uni") {
  discussionCategories(first:20) { nodes { id name } } } }'

# Create a discussion (body from a file)
jq -n --rawfile body <file.md> '{query: "mutation($repo: ID!, $cat: ID!, $title: String!,
  $body: String!) { createDiscussion(input: {repositoryId: $repo, categoryId: $cat,
  title: $title, body: $body}) { discussion { url number } } }",
  variables: {repo: "<repo-id>", cat: "<category-id>", title: "<title>", body: $body}}' \
  | gh api graphql --input -
```

### Pull requests

```bash
gh pr create --title "<title>" --body "<body>" --base main
gh pr list
gh pr view
```

PR gate before creating or updating a PR:

1. The final commit is done and `git status --short --branch` shows a clean tree.
2. The final head passes the local gate `make check` (ruff + pytest); record the
   result in the PR body.
3. The remote release workflow (`.github/workflows/release.yml`) runs only on
   `v*` tags and manual dispatch, so PRs are gated by the local checks plus
   review, not by remote CI.

### CI workflow runs

```bash
gh run list
gh run list --workflow=release.yml
gh run view <run-id>
gh run list --status=failure
```

## Release Process

- CI/release: `.github/workflows/release.yml`, triggered by `v*` tags or manual
  dispatch (with a `mujoco_version` input, default `3.11.0`).
- Pipeline: builds the sdist with `uv build --sdist`, installs it with
  `--no-build-isolation --no-binary`, and runs `python -m pytest tests/ -q` on a matrix
  of ubuntu (x64/arm), macOS, and Windows with Python 3.10 and 3.13. Only the sdist is
  published to PyPI (`skip-existing: true`).
