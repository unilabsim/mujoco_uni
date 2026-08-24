"""Tests for BatchEnvPool explicit worker CPU affinity (cpu_ids)."""

from __future__ import annotations

import os
import sys
from typing import Any

import mujoco
import numpy as np
import pytest

from mujoco_uni.batch_env import BatchEnvPool

mj: Any = mujoco

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="cpu_ids pinning is only supported on Linux"
)

PENDULUM_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" mass="1"/>
      <site name="tip" pos="0 0 -0.5"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge" gear="2"/>
  </actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <jointvel joint="hinge"/>
    <framepos objtype="site" objname="tip"/>
  </sensor>
</mujoco>
"""

# Module-level guard: os.sched_getaffinity does not exist off Linux, and the
# pytestmark skipif above only applies after collection evaluates this line.
_AVAILABLE_CPUS = sorted(os.sched_getaffinity(0)) if sys.platform == "linux" else []
_TWO_CPUS_REQUIRED = pytest.mark.skipif(
    len(_AVAILABLE_CPUS) < 2, reason="test needs at least 2 available CPUs"
)


def _model() -> mujoco.MjModel:
  return mj.MjModel.from_xml_string(PENDULUM_XML)


def _batched_states(model: mujoco.MjModel, nbatch: int) -> np.ndarray:
  nstate = mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)
  states = np.zeros((nbatch, nstate), dtype=np.float64)
  for i in range(nbatch):
    states[i, 0] = 0.05 * i
    states[i, model.nq] = 0.02 * i
  return states


def test_affinity_defaults_off() -> None:
  model = _model()
  with BatchEnvPool(model, nbatch=2, nthread=2) as pool:
    assert pool.cpu_ids is None
    assert pool.worker_cpu_ids() == ()


@_TWO_CPUS_REQUIRED
def test_cpu_ids_infers_nthread_and_pins_workers() -> None:
  cpu_ids = _AVAILABLE_CPUS[:2]
  with BatchEnvPool(_model(), nbatch=2, cpu_ids=cpu_ids) as pool:
    assert pool.nthread == 2
    assert pool.cpu_ids == tuple(cpu_ids)
    assert pool.worker_cpu_ids() == tuple(cpu_ids)


def test_cpu_ids_explicit_nthread_match() -> None:
  cpu_id = _AVAILABLE_CPUS[0]
  with BatchEnvPool(_model(), nbatch=2, nthread=1, cpu_ids=[cpu_id]) as pool:
    assert pool.nthread == 1
    assert pool.cpu_ids == (cpu_id,)
    assert pool.worker_cpu_ids() == (cpu_id,)


def test_cpu_ids_validation_errors() -> None:
  model = _model()
  cpu_id = _AVAILABLE_CPUS[0]

  with pytest.raises(ValueError, match="must equal nthread"):
    BatchEnvPool(model, nbatch=1, nthread=2, cpu_ids=[cpu_id])
  with pytest.raises(ValueError, match="must equal nthread"):
    BatchEnvPool(model, nbatch=1, nthread=0, cpu_ids=[cpu_id])
  with pytest.raises(ValueError, match="unique"):
    BatchEnvPool(model, nbatch=1, nthread=2, cpu_ids=[cpu_id, cpu_id])
  with pytest.raises(ValueError, match="non-empty"):
    BatchEnvPool(model, nbatch=1, cpu_ids=[])
  with pytest.raises(ValueError, match=">= 0"):
    BatchEnvPool(model, nbatch=1, cpu_ids=[-1])
  with pytest.raises(ValueError, match="not available"):
    BatchEnvPool(model, nbatch=1, cpu_ids=[max(_AVAILABLE_CPUS) + 4096])
  with pytest.raises(TypeError, match="integer"):
    BatchEnvPool(model, nbatch=1, cpu_ids=[1.5])
  with pytest.raises(TypeError, match="integer"):
    BatchEnvPool(model, nbatch=1, cpu_ids=[True])
  with pytest.raises(TypeError, match="sequence"):
    BatchEnvPool(model, nbatch=1, cpu_ids="0")


@_TWO_CPUS_REQUIRED
def test_pinned_pool_step_matches_unpinned() -> None:
  model = _model()
  nbatch = 4
  states = _batched_states(model, nbatch)

  with BatchEnvPool(model, nbatch=nbatch, nthread=2) as ref:
    expected = ref.step(states, nstep=5)
  with BatchEnvPool(model, nbatch=nbatch, cpu_ids=_AVAILABLE_CPUS[:2]) as pinned:
    got = pinned.step(states, nstep=5)

  np.testing.assert_array_equal(got, expected)
