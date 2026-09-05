# MuJoCoUni

简体中文 | [English](README.md)

<p align="center">
  <a href="https://arxiv.org/abs/2605.24922"><img src="https://img.shields.io/badge/arxiv-2605.24922-red" alt="arXiv"></a>
  <a href="https://unilabsim.github.io/paper/mujocouni.html"><img src="https://img.shields.io/badge/paper-MuJoCoUni-orange" alt="Paper"></a>
</p>

MuJoCoUni 是 UniLab 面向官方 MuJoCo 的独立批量执行器（batch-executor）层。它提供 UniLab 使用的 `BatchEnvPool` API，且不修改 MuJoCo 求解器、接触（contact）、积分器或源码树内部。

## 系统边界

MuJoCoUni 位于 UniLab 与官方 MuJoCo 之间。

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

职责划分：

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

对于更大规模的 CPU/GPU 训练，UniLab 仍然作为位于本地 MuJoCoUni 执行器之上的桥梁：

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

## 版本策略

MuJoCoUni 拥有自己的包版本，独立于 MuJoCo 求解器版本。

当前发布版本：

```text
mujoco-uni-runtime==0.5.0
mujoco>=3.5,<3.12
```

公开元数据可在 Python 中获取：

```python
import mujoco_uni

print(mujoco_uni.__version__)
print(mujoco_uni.MUJOCO_VERSION)
print(mujoco_uni.MUJOCO_VERSION_SPEC)
```

MuJoCoUni 在每个 Python 环境中仅支持一个官方 MuJoCo 求解器版本。原生扩（native extension）会记录构建时使用的 MuJoCo 版本，若运行时加载的 `mujoco` 包与该原生构建目标不匹配，会立即报错（fail fast）。

版本切换是生效的，但发生在进程层面：MuJoCoUni 会选择一个已有的带版本的 uv 环境，然后在 Python 导入 `mujoco` 之前，在该环境中运行目标命令。正常的训练启动不会创建、安装或重建环境。

```text
env-mj35  -> mujoco==3.5.x  -> build/install mujoco-uni-runtime
env-mj36  -> mujoco==3.6.x  -> build/install mujoco-uni-runtime
env-mj37  -> mujoco==3.7.x  -> build/install mujoco-uni-runtime
env-mj38  -> mujoco==3.8.x  -> build/install mujoco-uni-runtime
env-mj39  -> mujoco==3.9.x  -> build/install mujoco-uni-runtime
env-mj310 -> mujoco==3.10.x -> build/install mujoco-uni-runtime
env-mj311 -> mujoco==3.11.x -> build/install mujoco-uni-runtime
```

默认与回退（fallback）选择按如下顺序优先使用已发现的环境：

```text
3.11 > 3.8 > 3.10 > 3.9 > 3.7 > 3.6 > 3.5
```

如果未找到请求的版本，MuJoCoUni 会打印警告并回退到首选的现有环境。如果不存在任何 MuJoCo 环境，启动会以明确的配置错误而失败。

对于 UniLab 训练，常规任务命令是唯一面向用户的启动器。运行命令时设置进程选择器：

```bash
MUJOCO_UNI_VERSION=3.8 uv run train --algo ppo --task go2_joystick_flat --sim mujoco
MUJOCO_UNI_VERSION=3.10 uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

不设置 `MUJOCO_UNI_VERSION` 则保持当前激活的 Python 环境行为。诸如 `3.8.0` 的精确请求要求精确的运行时；诸如 `3.8` 的次版本（minor）请求接受兼容的 `3.8.x` 运行时。

MuJoCoUni 负责 UniLab 所使用的内部发现（discovery）与进程派生（spawning）服务。显式的环境准备属于配置（setup）操作，不属于正常的训练启动路径。

MuJoCo Python 模型指针所需的辅助函数 `_address` 和 `_from_model_ptr` 会在导入时进行检查。

## 包结构

其结构参照 DrakeUni 的运行时代码、原生源码与编译产物之间的划分：

```text
src/mujoco_uni/
  __init__.py
  metadata.py               # MuJoCoUni package metadata and supported range
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

稳定的导入路径为：

```python
from mujoco_uni.batch_env import BatchEnvPool, SUPPORTED_FIELDS
```

## 执行模型

`BatchEnvPool` 接受一个 `mujoco.MjModel`，或一个由`mujoco.MjModel` 对象组成的兼容序列。在构造时，它会从 Python 读取官方 MuJoCo 模型指针，用 `mj_copyModel` 拷贝模型，并在内部存储池（pool）持有的模型。

热路径（hot path）是原生 C++ 调用官方 MuJoCo C API。池使用：

- 每个环境槽位（slot）对应一个逻辑上的池持有 `mjModel` 赋值，
- 每个工作线程一个可复用的 `mjData`，
- 互不重叠的环境分块，
- 互不重叠的输出槽位，
- 每次批量操作一个同步点。

基础的 `BatchEnvPool` 执行器内部没有 MPI 或 OpenMP。大规模的多进程、多插槽（socket）或多节点采样，由位于 `BatchEnvPool`之上的一层组合多个本地执行器来完成。

### Worker CPU 亲和性（Linux）

`BatchEnvPool` 接受可选的 `cpu_ids` 序列，把每个原生工作线程钉到显式指定的 CPU 上，使多个池可以稳定在互不重叠的 CPU 集合上运行：

```python
pool = BatchEnvPool(model, nbatch=64, cpu_ids=[0, 1, 2, 3])
```

`cpu_ids[i]` 钉住工作线程 `i`，即与 `nthread` 一一对应：其长度必须等于 `nthread`；省略 `nthread` 时按 `len(cpu_ids)` 推断。CPU id 在构造时校验（非负、不重复、对当前进程可用），非法输入抛出 `ValueError`。配置的映射可通过只读属性 `pool.cpu_ids` 查询，`pool.worker_cpu_ids()` 返回钉核后每个 worker 实际观测到的 CPU。未配置 `cpu_ids` 时调度行为保持不变。CPU 钉核仅支持 Linux。

## 公共 API

稳定的公共 API 是 `mujoco_uni.batch_env`。

导出的符号：

- `BatchEnvPool`
- `SUPPORTED_FIELDS`
- `batch_available`
- `batch_import_error`

支持的随机化/模型字段：

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

`BatchEnvPool` 的核心行为：

- 由单个模型或长度为 `1` / `nbatch` 的序列构造，
- 对整个池执行 `step` 并返回最终状态，可选地附带最终传感器数据，
- 对整个池执行 `forward` 并返回传感器数据，
- 稀疏 `reset`，可选模型字段随机化（domain randomization），
- site 雅可比（Jacobian）查询，
- hfield 高度采样，
- 通过 `get_model`、`get_models` 和 `get_all_models` 获取非持有（non-owning）的模型视图。

返回的模型视图仅在池存活期间有效。

## 安装

MuJoCoUni 以 `mujoco-uni-runtime` 之名发布。默认安装路径是绑定默认 MuJoCo 版本（`3.11.0`）的**预编译 wheel**，覆盖 linux x86_64 / aarch64 与 macOS arm64 × Python 3.10–3.13：

```bash
pip install "mujoco==3.11.0" mujoco-uni-runtime
```

每个发布的运行时版本只携带一个 MuJoCo 绑定（PyPI 文件名唯一性约束），且原生扩展拒绝在构建时版本以外的任何 MuJoCo 版本下加载（导入时的精确版本看门狗）。

对于任何**非默认 MuJoCo 版本**，**sdist** 是兜底且唯一的路径：使用 `--no-build-isolation` 针对你环境中的 `mujoco` 从源码构建（避免构建发生在一次性的隔离环境中）。Windows 用户同样走此路径 —— 不提供 Windows wheel。

```bash
pip install "mujoco>=3.5,<3.12" pybind11 numpy setuptools wheel
pip install mujoco-uni-runtime --no-binary mujoco-uni-runtime --no-build-isolation
```

跨仓库的发布协调规则见 [docs/release-coordination.md](docs/release-coordination.md)。

### 前置要求

从源码构建需要：

- **C++17 工具链** —— macOS：Xcode Command Line Tools（`xcode-select --install`）；Debian/Ubuntu：`sudo apt install build-essential`；Windows：MSVC Build Tools
- **Python 开发头文件** —— 使用 Debian/Ubuntu 系统 Python 时：`sudo apt install python3-dev`（或与你的版本对应的 `python3.X-dev`）。使用 uv 托管的 Python（`uv python install`）则自带头文件，无需此步，也是推荐方式。

uv 项目以声明式达到同样效果：

```toml
[project.optional-dependencies]
mujoco = ["mujoco~=3.11.0", "mujoco-uni-runtime==0.5.0", "pybind11>=2.12", "wheel"]

[tool.uv]
no-build-isolation-package = ["mujoco-uni-runtime"]
```

在 UniLab 旁进行开发：

```bash
cd /path/to/mujoco_uni
uv sync
uv pip install --force-reinstall --no-deps --no-build-isolation -e .
```

常用开发命令也已封装在 `Makefile` 中：

```bash
make sync       # uv sync
make install    # editable 安装（编译原生扩展）
make check      # ruff + pytest
make test       # 仅运行 pytest
make matrix     # 跨受支持 MuJoCo 版本的版本矩阵检查
```

UniLab 通过其兼容/后端层导入，该层进而导入：

```python
from mujoco_uni.batch_env import BatchEnvPool, SUPPORTED_FIELDS
```

## 重建原生扩展

在单一环境中开发 MuJoCoUni 时，editable 安装很有用。它会生成本地扩展，例如：

```text
src/mujoco_uni/compiled/_batch_env.cpython-313-darwin.so
```

该产物与当前激活的 Python 环境、平台以及 MuJoCo 补丁版本绑定。切换虚拟环境、Python 版本或 MuJoCo 版本后需要重建：

```bash
uv pip install "mujoco==3.10.0" pybind11 wheel
uv pip install --force-reinstall --no-deps --no-build-isolation -e .
```

选择求解器版本后，如果想保留已构建的原生目标，运行本地检查时请避免自动 sync：

```bash
uv run --no-sync pytest -q
```

## 验证

独立检查：

```bash
uv run ruff check .
uv run pytest -q
```

或直接运行 `make check`。使用 `make mujoco MJ=<版本>` 切换 MuJoCo 版本后，
请用 `make test-no-sync` 运行测试，以保留刚构建的原生目标。

版本矩阵（version matrix）检查：

```bash
uv run python tools/version_matrix.py --pytest
```

或 `make matrix`。完整的 UniLab 验证可通过 `make test-unilab` 和
`make test-unilab-train` 运行。

默认矩阵覆盖：

```text
3.5.0 3.6.0 3.7.0 3.8.0 3.8.1 3.9.0 3.10.0 3.11.0
```

完整的 UniLab 任务验证独立于快速包矩阵：

```bash
uv run python tools/version_matrix.py --versions 3.5.0 3.10.0 --unilab
uv run python tools/version_matrix.py --versions 3.5.0 3.10.0 --unilab-train
```

`--unilab-train` 模式会在每个选定的 MuJoCo 环境中
运行一次简短的单迭代训练冒烟测试（smoke test）。

UniLab 集成检查：

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

训练冒烟测试：

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

## 线程安全

内置 MuJoCo 传感器从 `mjData.sensordata` 读取，批量执行器对其提供支持。

自定义传感器、插件（plugin）和 MuJoCo 全局回调对线程安全敏感。若 `nthread > 1`，任何全局可变的回调/插件状态均由调用方自行负责。

## 范围边界

MuJoCoUni 不承担以下职责：

- MuJoCo 源码修改，
- MuJoCo 求解器/接触/积分器的 fork，
- 将单个 MuJoCo 求解跨 MPI rank 分布式分解，
- 在 MuJoCoUni 执行器内部使用 OpenMP，
- UniLab 任务 YAML、rollout 代码、奖励函数以及 DrakeUni 行为。

## 引用

论文页面：[MuJoCoUni](https://unilabsim.github.io/paper/mujocouni.html)

```bibtex
@article{jia2026mujocouni,
  title   = {MuJoCoUni: Persistent Batched Runtime Primitives for MuJoCo},
  author  = {Jia, Yufei and Wu, Junzhe},
  journal = {arXiv preprint arXiv:2605.24922},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.24922}
}
```

## 许可证

Apache-2.0。
