# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Benchmark for BatchEnvPool.

Covers:
  * full-batch rollout throughput vs mujoco.rollout.rollout
  * reset_subset latency scan over nreset in {1, 8, 64, nbatch}
  * patch-only / refresh-only / patch+refresh latency isolation
  * memory footprint of the persistent model pool

Usage:
  python -m mujoco.batch_env_benchmark \
      --nbatch 1024 --nthread 16 --nstep 100 --warmup 3 --repeat 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gc
import time
from typing import List

import numpy as np

import mujoco
from mujoco import batch_env
from mujoco import rollout as stateless_rollout


BENCH_XML = r"""
<mujoco>
  <worldbody>
    <geom type="plane" size="10 10 .1"/>
    <body pos="0 0 .5">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" friction="0.8 0.005 0.0001"/>
      <body pos=".2 0 0">
        <joint axis="0 1 0"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0"/>
        <body pos=".3 0 0">
          <joint axis="0 1 0"/>
          <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# A 7-DOF serial arm with end-effector and elbow sites — a more
# representative target for site-Jacobian benchmarks (manipulation /
# end-effector control).
BENCH_XML_JAC = r"""
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .1">
      <joint name="j0" axis="0 0 1"/>
      <geom type="capsule" size=".03" fromto="0 0 0 0 0 .3"/>
      <body pos="0 0 .3">
        <joint name="j1" axis="0 1 0"/>
        <geom type="capsule" size=".03" fromto="0 0 0 .3 0 0"/>
        <site name="elbow" pos=".3 0 0"/>
        <body pos=".3 0 0">
          <joint name="j2" axis="1 0 0"/>
          <geom type="capsule" size=".03" fromto="0 0 0 .3 0 0"/>
          <body pos=".3 0 0">
            <joint name="j3" axis="0 1 0"/>
            <geom type="capsule" size=".03" fromto="0 0 0 .25 0 0"/>
            <body pos=".25 0 0">
              <joint name="j4" axis="1 0 0"/>
              <geom type="capsule" size=".025" fromto="0 0 0 .15 0 0"/>
              <body pos=".15 0 0">
                <joint name="j5" axis="0 1 0"/>
                <geom type="capsule" size=".02" fromto="0 0 0 .1 0 0"/>
                <body pos=".1 0 0">
                  <joint name="j6" axis="1 0 0"/>
                  <geom type="box" size=".04 .02 .02"/>
                  <site name="ee" pos=".04 0 0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _normalize_models(model_or_models, nbatch):
  if isinstance(model_or_models, mujoco.MjModel):
    return [model_or_models] * nbatch
  models = list(model_or_models)
  if len(models) == 1:
    return models * nbatch
  if len(models) != nbatch:
    raise ValueError(
        f"model sequence must have length 1 or nbatch={nbatch}, got {len(models)}"
    )
  return models


def _make_model_sequence(model, nbatch):
  models = [copy.copy(model) for _ in range(nbatch)]
  for env_id, env_model in enumerate(models):
    mass_scale = 1.0 + 0.01 * env_id
    friction_scale = 1.0 + 0.005 * env_id
    env_model.body_mass[:] *= mass_scale
    env_model.geom_friction[:] *= friction_scale
    data = mujoco.MjData(env_model)
    mujoco.mj_setConst(env_model, data)
  return models


def _time_it(fn, warmup: int, repeat: int) -> float:
  """Return median seconds per call."""
  for _ in range(warmup):
    fn()
  samples: List[float] = []
  for _ in range(repeat):
    t0 = time.perf_counter()
    fn()
    samples.append(time.perf_counter() - t0)
  return float(np.median(samples))


def bench_full_rollout(
    model_or_models, nbatch, nthread, nstep, warmup, repeat, rng, label
):
  models = _normalize_models(model_or_models, nbatch)
  nstate = mujoco.mj_stateSize(
      models[0], mujoco.mjtState.mjSTATE_FULLPHYSICS
  )
  state0 = rng.standard_normal((nbatch, nstate)).astype(np.float64)
  control = np.zeros((nbatch, nstep, models[0].nu), dtype=np.float64)

  # Stateless reference path — infers nthread from len(data).
  ref_models = [copy.copy(model) for model in models]
  ref_datas = (
      [mujoco.MjData(models[0]) for _ in range(nthread)]
      if nthread > 0
      else [mujoco.MjData(models[0])]
  )

  def ref_call():
    stateless_rollout.rollout(
        ref_models,
        ref_datas,
        state0,
        control,
    )

  t_ref = _time_it(ref_call, warmup, repeat)

  with batch_env.BatchEnvPool(
      model_or_models, nbatch=nbatch, nthread=nthread
  ) as pool:
    def pool_call():
      pool.step(
          state0,
          nstep=nstep,
          control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
          control=control,
      )
    t_pool = _time_it(pool_call, warmup, repeat)

  env_steps_per_sec_ref = nbatch * nstep / t_ref
  env_steps_per_sec_pool = nbatch * nstep / t_pool
  print(
      f"[full rollout:{label}] nbatch={nbatch} nthread={nthread} nstep={nstep}"
  )
  print(
      f"  stateless: {t_ref*1e3:7.2f} ms/call, "
      f"{env_steps_per_sec_ref/1e6:6.2f} Menv-steps/s"
  )
  print(
      f"  pool:      {t_pool*1e3:7.2f} ms/call, "
      f"{env_steps_per_sec_pool/1e6:6.2f} Menv-steps/s"
  )
  ratio = t_pool / t_ref if t_ref > 0 else float("nan")
  print(f"  pool/stateless ratio: {ratio:.3f}")


def bench_reset_subset_scan(
    model_or_models, nbatch, nthread, warmup, repeat, rng, label
):
  models = _normalize_models(model_or_models, nbatch)
  nstate = mujoco.mj_stateSize(
      models[0], mujoco.mjtState.mjSTATE_FULLPHYSICS
  )
  nresets = sorted(nreset for nreset in {1, 8, 64, nbatch} if nreset <= nbatch)

  with batch_env.BatchEnvPool(
      model_or_models, nbatch=nbatch, nthread=nthread
  ) as pool:
    print(f"[reset scan:{label}] nbatch={nbatch} nthread={nthread}")
    for nreset in nresets:
      env_ids = rng.choice(nbatch, size=nreset, replace=False).astype(np.int32)
      init = rng.standard_normal((nreset, nstate)).astype(np.float64)

      def call():
        pool.reset(env_ids, init)
      t = _time_it(call, warmup, repeat)
      print(
          f"  nreset={nreset:>5d}: {t*1e6:8.1f} us/call, "
          f"{t*1e6/max(nreset,1):6.2f} us/env"
      )


def bench_patch_refresh(model, nbatch, nthread, warmup, repeat, rng):
  nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  nreset = min(64, nbatch)

  with batch_env.BatchEnvPool(
      model, nbatch=nbatch, nthread=nthread
  ) as pool:
    env_ids = rng.choice(nbatch, size=nreset, replace=False).astype(np.int32)
    init = rng.standard_normal((nreset, nstate)).astype(np.float64)

    base_mass = pool.get_field(0, "body_mass")
    base_friction = pool.get_field(0, "geom_friction")
    new_mass = np.tile(base_mass, (nreset, 1))
    new_friction = np.tile(base_friction, (nreset, 1))

    scenarios = {
        "no_patch": None,
        "non_refresh_only (geom_friction)": {
            "geom_friction": new_friction
        },
        "refresh_only (body_mass)": {"body_mass": new_mass},
        "both (body_mass + geom_friction)": {
            "body_mass": new_mass,
            "geom_friction": new_friction,
        },
    }

    print(
        f"[patch+refresh isolation] nbatch={nbatch} nthread={nthread} "
        f"nreset={nreset}"
    )
    for label, rand in scenarios.items():
      def call():
        pool.reset(env_ids, init, randomization=rand)
      t = _time_it(call, warmup, repeat)
      print(f"  {label:38s}: {t*1e6:8.1f} us/call")


def _python_loop_site_jacobians(model, state0, site_ids, jacp, jacr):
  """Single-threaded Python reference: serial mj_jacSite over all envs."""
  data = mujoco.MjData(model)
  nbatch = state0.shape[0]
  K = len(site_ids)
  jp = np.empty((nbatch, K, 3, model.nv), dtype=np.float64) if jacp else None
  jr = np.empty((nbatch, K, 3, model.nv), dtype=np.float64) if jacr else None
  for r in range(nbatch):
    mujoco.mj_setState(
        model, data, state0[r], int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    )
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    for k, site in enumerate(site_ids):
      mujoco.mj_jacSite(
          model, data,
          jp[r, k] if jacp else None,
          jr[r, k] if jacr else None,
          int(site),
      )
  return jp, jr


def _python_threaded_site_jacobians(
    model, datas, state0, site_ids, jacp, jacr, executor
):
  """Python ThreadPoolExecutor reference. Each worker has its own mjData.

  Tests how much the GIL-released pybind helps Python-side parallelism.
  """
  nbatch = state0.shape[0]
  K = len(site_ids)
  jp = np.empty((nbatch, K, 3, model.nv), dtype=np.float64) if jacp else None
  jr = np.empty((nbatch, K, 3, model.nv), dtype=np.float64) if jacr else None
  nthread = len(datas)
  chunk = max(1, nbatch // (10 * nthread))

  def worker(thread_id, start, end):
    data = datas[thread_id]
    for r in range(start, end):
      mujoco.mj_setState(
          model, data, state0[r], int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
      )
      mujoco.mj_kinematics(model, data)
      mujoco.mj_comPos(model, data)
      for k, site in enumerate(site_ids):
        mujoco.mj_jacSite(
            model, data,
            jp[r, k] if jacp else None,
            jr[r, k] if jacr else None,
            int(site),
        )

  # Naive round-robin assignment of chunks to thread ids.
  futures = []
  job = 0
  start = 0
  while start < nbatch:
    end = min(start + chunk, nbatch)
    tid = job % nthread
    futures.append(executor.submit(worker, tid, start, end))
    start = end
    job += 1
  for f in futures:
    f.result()
  return jp, jr


def bench_jacobian(model, nbatch, K, nthread, warmup, repeat, rng, label):
  nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  state0 = rng.standard_normal((nbatch, nstate)).astype(np.float64)

  if model.nsite < 1:
    print(f"[jacobian:{label}] model has no sites — skipping")
    return
  K = min(K, model.nsite)
  site_ids = list(range(K))

  print(
      f"[jacobian:{label}] nbatch={nbatch} K={K} nthread={nthread} "
      f"nv={model.nv} nbody={model.nbody}"
  )

  # ---- baseline 1: pure-Python serial loop -----------------------------
  def py_serial():
    _python_loop_site_jacobians(model, state0, site_ids, True, False)
  t_py_serial = _time_it(py_serial, warmup, repeat)

  # ---- baseline 2: Python ThreadPoolExecutor ---------------------------
  if nthread > 0:
    py_datas = [mujoco.MjData(model) for _ in range(nthread)]
    py_executor = concurrent.futures.ThreadPoolExecutor(max_workers=nthread)
    try:
      def py_threaded():
        _python_threaded_site_jacobians(
            model, py_datas, state0, site_ids, True, False, py_executor
        )
      t_py_threaded = _time_it(py_threaded, warmup, repeat)
    finally:
      py_executor.shutdown(wait=True)
  else:
    t_py_threaded = float("nan")

  # ---- candidate: pool single-thread -----------------------------------
  with batch_env.BatchEnvPool(model, nbatch=nbatch, nthread=0) as pool_st:
    def pool_single():
      pool_st.compute_site_jacobians(state0, site_ids, jacp=True, jacr=False)
    t_pool_st = _time_it(pool_single, warmup, repeat)

  # ---- candidate: pool multi-thread ------------------------------------
  if nthread > 0:
    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool_mt:
      def pool_multi():
        pool_mt.compute_site_jacobians(state0, site_ids, jacp=True, jacr=False)
      t_pool_mt = _time_it(pool_multi, warmup, repeat)
  else:
    t_pool_mt = float("nan")

  # ---- numerical sanity (optional, cheap) ------------------------------
  jp_ref, _ = _python_loop_site_jacobians(model, state0, site_ids, True, False)
  with batch_env.BatchEnvPool(model, nbatch=nbatch, nthread=0) as pool_check:
    jp_pool, _ = pool_check.compute_site_jacobians(
        state0, site_ids, jacp=True, jacr=False
    )
  max_abs_err = float(np.max(np.abs(jp_ref - jp_pool)))

  rows = [
      ("python serial         ", t_py_serial),
      (f"python threadpool x{nthread} ", t_py_threaded),
      ("pool single-thread    ", t_pool_st),
      (f"pool multi-thread x{nthread}  ", t_pool_mt),
  ]
  base = t_py_serial
  for name, t in rows:
    if not np.isfinite(t):
      print(f"  {name}: n/a")
      continue
    speedup = base / t if t > 0 else float("nan")
    per_call_us = t * 1e6 / max(nbatch * K, 1)
    print(
        f"  {name}: {t*1e3:8.2f} ms  "
        f"({per_call_us:7.2f} us/(env*site))  speedup={speedup:5.1f}x"
    )
  print(f"  parity vs python-serial: max abs err = {max_abs_err:.2e}")


def bench_memory(model, nbatch):
  # Per-model memory via mj_sizeModel.
  bytes_per_model = mujoco.mj_sizeModel(model)
  total_mb = bytes_per_model * nbatch / (1024 * 1024)
  print(
      f"[memory] per-model size={bytes_per_model/1024:.1f} KiB, "
      f"nbatch={nbatch} → total model pool ≈ {total_mb:.1f} MiB"
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--nbatch", type=int, default=256)
  ap.add_argument("--nthread", type=int, default=4)
  ap.add_argument("--nstep", type=int, default=50)
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--repeat", type=int, default=5)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  rng = np.random.default_rng(args.seed)
  model = mujoco.MjModel.from_xml_string(BENCH_XML)
  model_sequence = _make_model_sequence(model, args.nbatch)
  jac_model = mujoco.MjModel.from_xml_string(BENCH_XML_JAC)
  print(
      f"model: nbody={model.nbody} nv={model.nv} "
      f"ngeom={model.ngeom} nsensordata={model.nsensordata}"
  )
  print(
      f"jac_model: nbody={jac_model.nbody} nv={jac_model.nv} "
      f"nsite={jac_model.nsite}"
  )

  bench_memory(model, args.nbatch)
  gc.collect()
  bench_full_rollout(
      model,
      args.nbatch,
      args.nthread,
      args.nstep,
      args.warmup,
      args.repeat,
      rng,
      "single-model",
  )
  gc.collect()
  bench_reset_subset_scan(
      model,
      args.nbatch,
      args.nthread,
      args.warmup,
      args.repeat,
      rng,
      "single-model",
  )
  gc.collect()
  bench_patch_refresh(
      model, args.nbatch, args.nthread, args.warmup, args.repeat, rng,
  )
  gc.collect()
  bench_full_rollout(
      model_sequence,
      args.nbatch,
      args.nthread,
      args.nstep,
      args.warmup,
      args.repeat,
      rng,
      "multi-model",
  )
  gc.collect()
  bench_reset_subset_scan(
      model_sequence,
      args.nbatch,
      args.nthread,
      args.warmup,
      args.repeat,
      rng,
      "multi-model",
  )
  gc.collect()

  # ---- Jacobian scan: small / medium / large -------------------------
  for label, jac_nbatch, K in (
      ("small",  64,         min(2, jac_model.nsite)),
      ("medium", 1024,       min(2, jac_model.nsite)),
      ("large",  args.nbatch, min(2, jac_model.nsite)),
  ):
    if jac_nbatch <= 0:
      continue
    bench_jacobian(
        jac_model, jac_nbatch, K, args.nthread,
        args.warmup, args.repeat, rng, label,
    )
    gc.collect()


if __name__ == "__main__":
  main()
