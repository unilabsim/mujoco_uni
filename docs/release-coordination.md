# Release Coordination Rules

Cross-repo rules for the MuJoCo 3.11 alignment roadmap
([unilabsim/UniLab#1515](https://github.com/unilabsim/UniLab/issues/1515)).
These rules govern how `mujoco-uni-runtime` releases interact with the
downstream consumers (`unisim`, `UniLab`) whenever the MuJoCo version moves.

## Version declaration layers

- **Consumer `pyproject.toml` bound** uses a compatible-release specifier:
  `mujoco~=3.11.0` (patch-flexible, minor-pinned).
- **Consumer `uv.lock`** pins the exact MuJoCo version (e.g. `3.11.0`).
- **Prebuilt wheels** bind the exact build-time MuJoCo (`3.11.0`): the native
  extension is compiled and linked against that `libmujoco` and records it as
  `MUJOCO_BUILD_VERSION`. The import-time exact-version watchdog refuses to
  load against any other MuJoCo patch version.

## One runtime release = one MuJoCo binding

PyPI identifies files by name; a `mujoco-uni-runtime` release cannot carry
wheels for several MuJoCo versions under one version number. Therefore each
released runtime version carries **exactly one** MuJoCo binding (the repo
default, currently `3.11.0`, `WHEEL_MUJOCO_VERSION` in
`.github/workflows/release.yml`).

Corollaries:

- **Any MuJoCo patch/minor default bump ⇒ `mujoco-uni-runtime` cuts a new
  release with new wheels.** There is no "same runtime version, new MuJoCo"
  path.
- **Downstream lock upgrades are gated on wheel availability**: `unisim` /
  `UniLab` only bump their pinned MuJoCo after the matching
  `mujoco-uni-runtime` release (with wheels) is published.
- **Switching MuJoCo versions locally always uses the sdist rebuild path**:
  `make mujoco MJ=<version>` (or `pip install --no-binary mujoco-uni-runtime`
  with `--no-build-isolation`). The runtime watchdog error message points
  there.

## Wheel matrix

Prebuilt wheels follow the supported Python × platform matrix and evolve
with it:

- Python: 3.10, 3.11, 3.12, 3.13
- Platforms: linux x86_64 (`ubuntu-latest`), linux aarch64
  (`ubuntu-24.04-arm`), darwin arm64 (`macos-latest`)

= 12 wheels per release, all built against `mujoco==3.11.0` with
`--no-build-isolation`, smoke-tested in CI before upload. Windows wheels are
out of scope; Windows users install the sdist, which is exercised on
`windows-latest` in the sdist test matrix.

## Release flow

1. `workflow_dispatch` runs the full sdist + wheel build/verify matrix
   without publishing — use it to verify a head SHA before tagging.
2. Tagging `v*` reruns the same matrix and, on success, publishes the sdist
   plus all 12 wheels to PyPI via trusted publishing (OIDC,
   `environment: pypi`, `skip-existing: true`).
3. Only after the release is visible on PyPI may downstream repos bump their
   locks.
