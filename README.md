# MuJoCoUni

English | [简体中文](README_zh.md)

<p align="center">
  <a href="https://arxiv.org/abs/2605.24922"><img src="https://img.shields.io/badge/arxiv-2605.24922-red" alt="arXiv"></a>
  <a href="https://unilabsim.github.io/paper/mujocouni.html"><img src="https://img.shields.io/badge/paper-MuJoCoUni-orange" alt="Paper"></a>
</p>

MuJoCoUni is the standalone UniLab batch-executor layer for official MuJoCo.
It provides the `BatchEnvPool` API used by UniLab without modifying MuJoCo
solver, contact, integrator, or source-tree internals.

## System Boundary

MuJoCoUni sits between UniLab and official MuJoCo.

```text
MJCF / XML asset
        |
        v
official MuJoCo compiler
        |
        v
mjModel
        |
        v
MuJoCoUni BatchEnvPool
  - captures/copies mjModel
  - owns model pool
  - owns worker mjData
  - calls official MuJoCo C APIs
        ^
        |
UniLab backend
  - receives task/training command
  - owns rollout pace
  - packs state/control/reset arrays
  - unpacks state/sensor arrays
```

Responsibility split:

```text
UniLab:
  task config, commands, rewards, rollout pace, CPU/GPU bridge,
  training orchestration, logging

MuJoCoUni:
  BatchEnvPool, model cloning, worker mjData, local thread pool,
  batched step/forward/reset/query execution

MuJoCo:
  MJCF compiler, mjModel/mjData definitions, solver, contact,
  integrator, sensor layout, official C API
```

For larger CPU/GPU training, UniLab remains the bridge above local MuJoCoUni
executors:

```text
CPU rollout side
  MuJoCoUni BatchEnvPool instances
  produce observation/reward/done/sensor data
  request action tensors
        |
        v
UniLab bridge
  batches rollout traffic
  routes data toward GPU learner/action service
  routes actions/control commands back to CPU workers
        |
        v
GPU side
  consumes training data
  performs learner/inference work
  returns action tensors or policy-side feedback through UniLab
```

## Version Policy

MuJoCoUni has its own package version, independent of the MuJoCo solver version.

Current release:

```text
mujoco-uni-runtime==0.4.0
mujoco>=3.5,<3.11
```

The public metadata is available from Python:

```python
import mujoco_uni

print(mujoco_uni.__version__)
print(mujoco_uni.MUJOCO_VERSION)
print(mujoco_uni.MUJOCO_VERSION_SPEC)
```

MuJoCoUni supports one official MuJoCo solver version per Python environment.
The native extension records the MuJoCo version used at build time, and the
runtime fails fast if the loaded `mujoco` package does not match that native
build target.

Version switching is active but process-level: MuJoCoUni selects an existing
versioned uv environment, then runs the target command in that environment
before Python imports `mujoco`. Normal training launch does not create, install,
or rebuild environments.

```text
env-mj35  -> mujoco==3.5.x  -> build/install mujoco-uni-runtime
env-mj36  -> mujoco==3.6.x  -> build/install mujoco-uni-runtime
env-mj37  -> mujoco==3.7.x  -> build/install mujoco-uni-runtime
env-mj38  -> mujoco==3.8.x  -> build/install mujoco-uni-runtime
env-mj39  -> mujoco==3.9.x  -> build/install mujoco-uni-runtime
env-mj310 -> mujoco==3.10.x -> build/install mujoco-uni-runtime
```

Default and fallback selection prefer discovered environments in this order:

```text
3.8 > 3.10 > 3.9 > 3.7 > 3.6 > 3.5
```

If the requested version is not found, MuJoCoUni prints a warning and falls back
to the preferred existing environment. If no MuJoCo environment exists, launch
fails with a clear setup error.

For UniLab training, the normal task command is the only user-facing launcher.
Set the process selector when running the command:

```bash
MUJOCO_UNI_VERSION=3.8 uv run train --algo ppo --task go2_joystick_flat --sim mujoco
MUJOCO_UNI_VERSION=3.10 uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

Unset `MUJOCO_UNI_VERSION` keeps the active Python environment behavior. Exact
requests such as `3.8.0` require that exact runtime; minor requests such as
`3.8` accept a compatible `3.8.x` runtime.

MuJoCoUni owns the internal discovery and spawning services used by UniLab.
Explicit environment preparation is a setup operation, not part of the normal
training launch path.

The required MuJoCo Python model pointer helpers, `_address` and
`_from_model_ptr`, are checked at import time.

## Roadmap

Further development is tracked in [ROADMAP.md](ROADMAP.md).

## Package Layout

The structure mirrors the DrakeUni split between runtime code, native source,
and compiled artifacts:

```text
src/mujoco_uni/
  __init__.py
  metadata.py              # MuJoCoUni package metadata and supported range
  batch_env.py              # Stable public BatchEnvPool API
  mujoco_runtime/
    api.py                  # MuJoCoUni-owned access to official mujoco
    version_control.py      # MuJoCo solver-version control
  runtime/
    batch.py                # Python API, validation, compatibility behavior
  compiled/
    __init__.py             # Native extension loader and diagnostics
    _batch_env*.so          # Generated extension after local build
  native/
    batch_env.cc            # Native pybind11 executor
    threadpool.h/.cc        # Local thread pool
```

The stable import path is:

```python
from mujoco_uni.batch_env import BatchEnvPool, SUPPORTED_FIELDS
```

## Execution Model

`BatchEnvPool` accepts one `mujoco.MjModel` or a compatible sequence of
`mujoco.MjModel` objects. At construction time it reads the official MuJoCo
model pointer from Python, copies the model with `mj_copyModel`, and stores
pool-owned models internally.

The hot path is native C++ calling the official MuJoCo C API. The pool uses:

- one logical pool-owned `mjModel` assignment per environment slot,
- one reusable `mjData` per worker thread,
- disjoint environment chunks,
- disjoint output slots,
- one synchronization point per batch operation.

There is no MPI or OpenMP inside the base `BatchEnvPool` executor. Large-scale
multi-process, multi-socket, or multi-node collection composes multiple local
executors from a layer above `BatchEnvPool`.

### Worker CPU affinity (Linux)

`BatchEnvPool` accepts an optional `cpu_ids` sequence that pins each native
worker thread to an explicit CPU, so multiple pools can be kept on disjoint
CPU sets:

```python
pool = BatchEnvPool(model, nbatch=64, cpu_ids=[0, 1, 2, 3])
```

`cpu_ids[i]` pins worker thread `i`, so the mapping is 1:1 with `nthread`:
its length must equal `nthread`, and when `nthread` is omitted it is inferred
from `len(cpu_ids)`. CPU ids are validated at construction (non-negative,
unique, available to the process); invalid input raises `ValueError`. The
configured mapping is exposed read-only via `pool.cpu_ids`, and
`pool.worker_cpu_ids()` reports the CPU each worker was actually observed on
after pinning. When `cpu_ids` is not given, scheduling behavior is unchanged.
CPU pinning is supported on Linux only.

## Public API

The stable public API is `mujoco_uni.batch_env`.

Exported symbols:

- `BatchEnvPool`
- `SUPPORTED_FIELDS`
- `batch_available`
- `batch_import_error`

Supported randomization/model fields:

```text
body_mass
body_ipos
body_iquat
body_inertia
dof_armature
gravity
geom_friction
kp
kd
```

Core `BatchEnvPool` behavior:

- construct from one model or a sequence of length `1` / `nbatch`,
- `step` over the full pool and return final state, optionally with final
  sensor data,
- `forward` over the full pool and return sensor data,
- sparse `reset` with optional model-field randomization,
- site Jacobian queries,
- hfield height sampling,
- non-owning model views through `get_model`, `get_models`, and
  `get_all_models`.

Returned model views remain valid only while the pool is alive.

## Installation

MuJoCoUni is published as `mujoco-uni-runtime` (**source distribution only**):

```bash
pip install "mujoco>=3.5,<3.11" pybind11 numpy setuptools wheel
pip install mujoco-uni-runtime --no-build-isolation
```

There are no prebuilt wheels on purpose: the native extension is compiled
against the `mujoco` package present at build time and refuses to load against
any other MuJoCo version, so a prebuilt wheel would silently bind you to one
MuJoCo release. Install with `--no-build-isolation` (as above) so the
extension is compiled against the `mujoco` version of your target environment
instead of a throwaway isolated one.

### Prerequisites

Building from source requires:

- a **C++17 toolchain** — macOS: Xcode Command Line Tools (`xcode-select --install`);
  Debian/Ubuntu: `sudo apt install build-essential`; Windows: MSVC Build Tools
- **Python development headers** — Debian/Ubuntu system Python:
  `sudo apt install python3-dev` (or `python3.X-dev` matching your version).
  Not needed when using a uv-managed Python (`uv python install`), which
  bundles the headers and is the recommended option.

uv projects declare the same setup:

```toml
[project.optional-dependencies]
mujoco = ["mujoco>=3.5,<3.11", "mujoco-uni-runtime==0.4.0", "pybind11>=2.12", "wheel"]

[tool.uv]
no-build-isolation-package = ["mujoco-uni-runtime"]
```

For development beside UniLab:

```bash
cd /path/to/mujoco_uni
uv sync
uv pip install --force-reinstall --no-deps --no-build-isolation -e .
```

UniLab imports through its compatibility/backend layer, which in turn imports:

```python
from mujoco_uni.batch_env import BatchEnvPool, SUPPORTED_FIELDS
```

## Rebuilding The Native Extension

Editable installs are useful while developing MuJoCoUni in one environment.
They generate a local extension such as:

```text
src/mujoco_uni/compiled/_batch_env.cpython-313-darwin.so
```

That artifact is tied to the active Python environment, platform, and MuJoCo
patch version. Rebuild after switching virtual environments, Python versions, or
MuJoCo versions:

```bash
uv pip install "mujoco==3.10.0" pybind11 wheel
uv pip install --force-reinstall --no-deps --no-build-isolation -e .
```

After selecting a solver version, run local checks without an automatic sync if
you want to preserve the already-built native target:

```bash
uv run --no-sync pytest -q
```

## Validation

Standalone checks:

```bash
uv run ruff check .
uv run pytest -q
```

Version-matrix checks:

```bash
uv run python tools/version_matrix.py --pytest
```

The default matrix covers:

```text
3.5.0 3.6.0 3.7.0 3.8.0 3.8.1 3.9.0 3.10.0
```

Full UniLab task validation is separate from the quick package matrix:

```bash
uv run python tools/version_matrix.py --versions 3.5.0 3.10.0 --unilab
uv run python tools/version_matrix.py --versions 3.5.0 3.10.0 --unilab-train
```

The `--unilab-train` mode runs a short one-iteration training smoke in each
selected MuJoCo environment.

UniLab integration checks:

```bash
cd ../UniLab
uv run pytest \
  tests/base/test_mujoco_batch_env_randomization.py \
  tests/base/test_mujoco_batch_env_jacobian.py \
  tests/base/backend/test_mujoco_site_jacobian.py \
  tests/envs/locomotion/test_go2_rough_height_scan.py \
  tests/envs/locomotion/test_go2_footstand.py \
  tests/envs/locomotion/test_go2_terrain_spawn.py \
  -q
```

Training smoke:

```bash
cd ../UniLab
uv run python scripts/train_rsl_rl.py \
  task=go2_joystick_flat/mujoco \
  algo.seed=1 \
  algo.num_envs=256 \
  algo.num_steps_per_env=24 \
  algo.max_iterations=2 \
  algo.save_interval=100 \
  training.no_play=true \
  training.logger=tensorboard \
  training.device=cpu \
  training.log_root=logs/mujoco_uni_smoke
```

## Thread Safety

Built-in MuJoCo sensors are read from `mjData.sensordata` and are supported by
the batch executor.

Custom sensors, plugins, and global MuJoCo callbacks are thread-safety-sensitive.
If `nthread > 1`, any global mutable callback/plugin state is the caller's
responsibility.

## Scope Boundaries

MuJoCoUni stays outside these responsibilities:

- MuJoCo source-code changes,
- MuJoCo solver/contact/integrator forks,
- distributed factorization of one MuJoCo solve across MPI ranks,
- OpenMP inside the MuJoCoUni executor,
- UniLab task YAMLs, rollout code, reward functions, and DrakeUni behavior.

## Citation

Paper page: [MuJoCoUni](https://unilabsim.github.io/paper/mujocouni.html)

```bibtex
@article{jia2026mujocouni,
  title   = {MuJoCoUni: Persistent Batched Runtime Primitives for MuJoCo},
  author  = {Jia, Yufei and Wu, Junzhe},
  journal = {arXiv preprint arXiv:2605.24922},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.24922}
}
```

## License

Apache-2.0.
