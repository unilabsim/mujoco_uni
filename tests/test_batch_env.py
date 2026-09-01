from __future__ import annotations

import importlib.metadata
import re
from typing import Any

import mujoco

import mujoco_uni


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    assert match is not None
    return tuple(int(match.group(i)) for i in range(1, 4))


def test_package_version_is_independent_from_solver_version() -> None:
    assert importlib.metadata.version("mujoco-uni-runtime") == mujoco_uni.__version__
    assert mujoco_uni.__version__ == "0.4.0"
    assert mujoco_uni.MUJOCO_DEFAULT_VERSION == "3.8.0"
    assert mujoco_uni.MUJOCO_MIN_VERSION == "3.5.0"
    assert mujoco_uni.MUJOCO_MAX_VERSION_EXCLUSIVE == "3.11.0"
    assert mujoco_uni.MUJOCO_VERSION_SPEC == ">=3.5,<3.11"
    assert (3, 5, 0) <= _version_tuple(mujoco.__version__) < (3, 11, 0)


def test_batch_env_constructs_from_official_mujoco_model() -> None:
    from mujoco_uni.batch_env import SUPPORTED_FIELDS, BatchEnvPool
    from mujoco_uni.compiled import NativeBatchEnvPool, batch_available, batch_import_error
    from mujoco_uni.runtime import available_backends, batch_diagnostics

    mj: Any = mujoco
    model = mj.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="box">
              <freejoint/>
              <geom type="box" size="0.1 0.1 0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    assert batch_available()
    assert batch_import_error() is None
    assert available_backends() == {"batch": True}
    assert batch_diagnostics()["batch_import_error"] is None
    assert mujoco_uni.BatchEnvPool is BatchEnvPool
    assert NativeBatchEnvPool is not BatchEnvPool
    assert set(SUPPORTED_FIELDS) == {
        "body_mass",
        "body_ipos",
        "body_iquat",
        "body_inertia",
        "dof_armature",
        "gravity",
        "geom_friction",
        "kp",
        "kd",
    }

    with BatchEnvPool(model, nbatch=2, nthread=1) as pool:
        assert pool.nbatch == 2
        assert pool.nthread == 1
        assert pool.nstate == mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)
        assert pool.get_model(0).nbody == model.nbody


_BLOWUP_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="arm">
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.005 0.05" fromto="0 0 0 0.1 0 0" mass="1e-6"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m0" joint="j0" gear="1e9" ctrlrange="-1e9 1e9"/>
  </actuator>
</mujoco>
"""


def test_step_exposes_per_env_autoreset_mask() -> None:
    import numpy as np

    from mujoco_uni.batch_env import BatchEnvPool

    model = mujoco.MjModel.from_xml_string(_BLOWUP_XML)
    for nthread in (0, 2):
      with BatchEnvPool(model, nbatch=4, nthread=nthread) as pool:
        state = np.zeros((4, pool.nstate), dtype=np.float64)
        state[:, 1] = 0.1
        control = np.zeros((4, model.nu), dtype=np.float64)
        control[2, 0] = 1e9

        assert pool.was_autoreset.dtype == np.bool_
        assert not pool.was_autoreset.any()
        out = pool.step(state, nstep=3, control=control)
        assert pool.was_autoreset.tolist() == [False, False, True, False]
        assert out[2, 1] == 0.0

        pool.step(state, nstep=3, control=np.zeros_like(control))
        assert not pool.was_autoreset.any()
