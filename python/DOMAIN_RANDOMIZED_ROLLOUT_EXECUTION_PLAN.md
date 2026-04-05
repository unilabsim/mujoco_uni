# Domain-Randomized Rollout Execution Plan

This document defines the execution plan for adding high-performance
reset-lifecycle domain randomization support to the MuJoCo Python bindings.

It is intentionally scoped to the `python/` bindings layer and the C++ pybind
modules that back it. It is not a consumer-side integration plan for UniLab or
any other environment package.


## 1. Problem Statement

The current Python bindings already support:

- full-batch rollout over one shared `MjModel`
- full-batch rollout over `nbatch` distinct compatible `MjModel` instances
- batched `mj_forward` through `batch_forward`

That is enough to express the semantics of per-environment model variation, but
it is not enough to deliver a production-quality reset-lifecycle domain
randomization path for large RL workloads.

The missing piece is a persistent executor that owns:

- a per-environment `mjModel` pool
- a per-thread `mjData` pool
- subset reset forward
- batched model patching
- selective `mj_setConst`

Without that executor, the control plane is forced to reassemble the hot path in
Python, which is not acceptable for large CPU rollout workloads.


## 2. Scope

This plan covers:

- CPU rollout only
- reset-lifecycle model randomization
- per-environment persistent `mjModel` pools
- subset reset fast paths
- selective `mj_setConst` refresh
- a small initial set of model fields

This plan does not cover:

- step-lifecycle domain randomization
- bucket models or `env_to_model_idx`
- public documentation changes in `doc/`
- a declarative event system
- consumer-side manager abstractions
- full field coverage on day one


## 3. Hard Requirements

The implementation must satisfy all of the following.

### 3.1 Data-plane work must stay out of Python

The following are not acceptable as the main implementation path:

- per-environment Python patch loops
- per-environment Python `mj_setConst` loops
- Python-side subset model gathering on every reset
- Python-side scans over all `nbatch` models when `nreset << nbatch`

Python may still orchestrate lifecycle decisions and construct payloads, but the
executor path itself must run in C++.

### 3.2 The rollout stepping model must remain unchanged

The implementation must reuse the current stepping structure:

- one model per environment
- one data object per worker thread
- current threaded rollout semantics

This work is about model ownership, reset execution, and selective refresh. It
is not about redesigning MuJoCo's threaded stepping path.

### 3.3 The implementation must support sparse resets

When only a subset of environments resets, the executor must:

- patch only those environments
- refresh only those environments that hit refresh fields
- forward only those environments
- write back only those environments

Any design that falls back to full-batch work for `nreset = 1` is a failure.

### 3.4 Runtime model cloning must not go through temporary files

The production runtime path must use in-memory cloning with `mj_copyModel`.

Using `mj_saveModel` and `mj_loadModel` or Python `.mjb` round-trips is useful
for experiments and benchmarks, but it is not an acceptable final executor
implementation.


## 4. Evaluated Options

Three implementation directions were considered.

### 4.1 Option A: implement the first version in the consumer backend

This means putting model pools, subset reset, patching, and refresh logic in a
consumer package such as UniLab.

Decision:

- reject

Reason:

- the hot path ends up in Python
- the abstraction boundary becomes consumer-specific
- the bindings layer remains missing the actual execution primitive

### 4.2 Option B: only extend `rollout` and `batch_forward`

This means continuing to use the existing stateless wrapper shape and only
adding more parameters to `_rollout` and `_batch_forward`.

Decision:

- not the primary plan

Reason:

- it helps with model selection, but it still does not provide a persistent
  executor object
- model pool ownership, patching, refresh, and subset reset orchestration still
  tend to leak back to Python
- the result is harder to keep coherent as field coverage grows

This option may still be useful later for auxiliary optimizations, but it
should not be the first implementation target.

### 4.3 Option C: add a dedicated persistent executor module

This means adding a new pybind-backed module under `python/mujoco/` that owns
the model pool, worker data, and reset execution path.

Decision:

- adopt

Reason:

- it keeps the data plane in C++
- it cleanly separates rollout execution from consumer orchestration
- it scales better to sparse reset randomization
- it does not overload the current stateless wrapper APIs with stateful runtime
  responsibilities


## 5. Chosen Architecture

Add a new Python submodule and a new C++ pybind module:

- `python/mujoco/domain_randomized_rollout.py`
- `python/mujoco/domain_randomized_rollout.cc`
- `_domain_randomized_rollout` entry in `python/mujoco/CMakeLists.txt`

Suggested Python-facing object name:

- `DomainRandomizedRolloutPool`

This object should own:

- one immutable input `base_model`
- `models_[nbatch]`, cloned from `base_model`
- `worker_data_[nthread]`
- a thread pool
- metadata needed for patching and refresh

This object should provide two public execution paths:

1. `rollout(...)`
2. `reset_subset(...)`

`reset_subset(...)` is the important one. It is the endpoint that will
eventually fuse:

- subset model patching
- selective `mj_setConst`
- subset `mj_setState`
- subset `mj_forward`
- state and sensor write-back


## 6. Why a Dedicated Module Is Better Than Extending `rollout`

The existing `rollout` and `batch_forward` modules are stateless kernels:

- they receive models and data
- they execute
- they return outputs

The proposed reset randomization path is stateful by construction:

- the model pool must persist across episodes
- patches apply to persistent per-environment models
- refresh is selective and field-dependent
- rollout and reset operate over the same owned model pool

Trying to force that design into the current wrapper shape would either:

- push state back to Python, or
- make the existing wrappers significantly harder to reason about

A dedicated executor object keeps the new responsibilities local.


## 7. API Direction

This document defines the intended executor shape, not a frozen public API.
The initial module should be treated as experimental until field coverage and
benchmarks stabilize.

The intended Python surface is:

```python
pool = domain_randomized_rollout.DomainRandomizedRolloutPool(
    model=base_model,
    nbatch=nbatch,
    nthread=nthread,
)

state, sensordata = pool.rollout(
    initial_state=initial_state,
    control=control,
    control_spec=...,
    initial_warmstart=...,
)

state_subset, sensordata_subset = pool.reset_subset(
    env_ids=env_ids,
    initial_state=state_subset,
    initial_warmstart=warmstart_subset,
    randomization=randomization_plan,
    skipsensor=False,
)
```

The important part is not the exact Python signature. The important part is that
`reset_subset` remains a fused executor entry point, not a Python script that
chains several lower-level calls together.


## 8. Field Coverage Strategy

The executor must support incremental field coverage. It should not assume that
all useful model-level randomization fields will land in the first iteration.

The initial field set should be:

- `body_mass`
- `body_ipos`
- `geom_friction`
- `dof_armature`
- optionally `dof_damping`

Refresh groups:

- requires `mj_setConst`:
  - `body_mass`
  - `body_ipos`
  - `dof_armature`
- does not initially require `mj_setConst`:
  - `geom_friction`
  - `dof_damping`

Deferred fields:

- `body_inertia`
- `body_iquat`
- `actuator_gainprm`
- `actuator_biasprm`
- `qpos0`

These should only be added after the executor path is stable and benchmarked.


## 9. Reset Semantics

This plan is strictly for reset-lifecycle model randomization.

That means:

- a model randomization sample is taken when an environment resets
- that sample remains fixed for the entire episode
- non-reset environments do not participate in that update

This plan explicitly does not cover model updates during ordinary stepping.


## 10. Memory Policy

The executor does not own XML preprocessing policy. It should accept an already
constructed `MjModel` and clone from that model.

However, for large per-environment model pools, callers that enable model-level
domain randomization should strongly prefer models compiled with
`compiler.discardvisual=true`.

That is a caller-side model construction rule, not an executor-side XML rewrite
feature.


## 11. Concrete Implementation Plan

### Phase 0: baseline and repo plumbing

Goal:

- establish a clean implementation target and validation baseline

Work:

- add this document
- identify current rollout and batch-forward parity tests
- record baseline throughput and memory results before introducing the new
  executor

Deliverables:

- this plan
- baseline measurements

### Phase 1: add the new module skeleton

Goal:

- add a new pybind module and Python wrapper with no randomization support yet

Files:

- `python/mujoco/CMakeLists.txt`
- `python/mujoco/domain_randomized_rollout.cc`
- `python/mujoco/domain_randomized_rollout.py`

Requirements:

- the module builds cleanly
- the Python wrapper exposes a pool object
- the object owns `nbatch` cloned models using `mj_copyModel`
- the object owns `nthread` worker data instances

Non-goals:

- no patching
- no refresh
- no public documentation updates in `doc/`

### Phase 2: full-batch rollout parity on owned model pools

Goal:

- verify that the executor can replace the current full-batch rollout path when
  all models are identical

Requirements:

- `rollout(...)` matches existing `mujoco.rollout.rollout(...)`
- no output regressions against current rollout behavior
- no unreasonable throughput regression for the identical-model case

Tests to add:

- parity against current rollout for one shared model
- parity against current rollout for `nbatch` cloned identical models

### Phase 3: subset reset forward without randomization

Goal:

- add the sparse reset execution primitive without any model patching

Requirements:

- `reset_subset(...)` accepts `env_ids`
- it only processes the subset
- it returns state and sensor outputs for the subset only
- it does not fall back to full-batch work

Tests to add:

- empty subset is a fast no-op
- one-environment subset only changes one output row
- many-environment subsets match explicit reference forwards

### Phase 4: model patching and selective refresh

Goal:

- add reset-lifecycle model randomization to the subset reset path

Requirements:

- only the addressed environments are patched
- only refresh fields trigger `mj_setConst`
- non-refresh fields do not trigger `mj_setConst`
- patch + refresh + subset forward stay inside the fused C++ path

Tests to add:

- target environments change, non-target environments do not
- refresh fields call the refresh path only for touched environments
- sparse field payloads leave absent fields untouched

### Phase 5: benchmark and profiling pass

Goal:

- ensure the executor is actually viable for large sparse-reset workloads

Metrics to capture:

- full-batch rollout throughput
- subset reset latency
- patch latency
- refresh latency
- memory cost of the persistent pool

Stop conditions:

- Python orchestration remains a dominant cost
- subset reset still scales with `nbatch` rather than `nreset`
- full-batch rollout regresses unacceptably for identical-model pools

### Phase 6: consumer integration

Goal:

- integrate the new executor into a consumer backend such as UniLab

Important note:

- this phase happens outside this repository
- consumer packages should treat this module as the data plane
- consumer packages should not rebuild patch and refresh logic in Python


## 12. Suggested Test Files

Add a new dedicated test file:

- `python/mujoco/domain_randomized_rollout_test.py`

This file should cover:

- constructor behavior
- clone compatibility
- rollout parity
- subset reset behavior
- selective refresh behavior
- sparse payload semantics

The existing files remain useful as regression guards:

- `python/mujoco/rollout_test.py`
- `python/mujoco/bindings_test.py`


## 13. Suggested PR Split

The work should be split into small, reviewable changes.

### PR 1

- add module skeleton
- add model pool ownership
- add rollout parity tests

### PR 2

- add no-randomization `reset_subset`
- add subset reset tests

### PR 3

- add the initial patch field set
- add selective refresh
- add field-level tests

### PR 4

- add benchmarks
- add follow-up cleanup and documentation notes if the API is considered stable


## 14. Explicit Non-Goals for the First Iteration

Do not spend the first iteration on:

- bucketed model APIs
- `env_to_model_idx`
- generalized event systems
- upstream public documentation pages
- trying to solve every domain-randomization field at once

The first iteration succeeds if it delivers a clean, benchmarked, reset-focused
executor with a small but defensible field set.


## 15. Final Decision

The correct first implementation target is:

- a dedicated persistent executor in the MuJoCo Python bindings

The correct first public capability is:

- full-batch rollout on an owned per-environment model pool
- fused subset reset for reset-lifecycle model randomization

The wrong first implementation target is:

- consumer-side Python backend code

The wrong first optimization target is:

- bucket models or generalized model-index indirection
