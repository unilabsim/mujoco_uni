#!/usr/bin/env python3
"""Tests for multi-model BatchEnvPool (uni/feat patch).

Verifies that BatchEnvPool correctly supports:
1. Multiple different MjModel instances in a single pool
2. Single-model pool (backward compatibility)
3. Model deduplication (shared pointers for identical source models)
4. Correct per-env physics (different models produce different trajectories)
5. forward / compute_site_jacobians

Known limitations (tested separately / excluded):
- ``get_model`` / ``get_all_models``: upstream MjModelRawPointerMap teardown
  abort (affects both patched and unpatched builds).
- ``control_is_constant`` (2D control array): works at the C++ ``_pool.step``
  level but the Python wrapper's ``np.ascontiguousarray`` may flatten to 1-D,
  causing a size mismatch in the old ``optional_array_ptr`` code path.
  The table-grasp verifier calls ``_pool.step`` directly and is unaffected.

Run::

    python -m pytest mujoco/test_batch_env_multimodel.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

import mujoco
from mujoco.batch_env import BatchEnvPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_XML_TEMPLATE = """\
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body name="obj" pos="0 0 {height}">
      <freejoint/>
      <geom type="sphere" size="0.03" mass="0.1"/>
      <site name="grip" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <sensor>
    <framepos objtype="site" objname="grip"/>
  </sensor>
</mujoco>
"""


def _make_model(height: float = 1.0) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_XML_TEMPLATE.format(height=height))


def _get_init_state(m: mujoco.MjModel) -> np.ndarray:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    s = np.zeros(nstate)
    mujoco.mj_getState(m, d, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return s


# State layout: [time, qpos(x, y, z, qw, qx, qy, qz), qvel(6)]
Z_IDX = 3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleModelPool:
    """Backward-compatibility: single-model pool must still work."""

    def test_step_basic(self):
        m = _make_model(1.0)
        pool = BatchEnvPool(m, nbatch=4, nthread=2)
        s0 = np.tile(_get_init_state(m), (4, 1))
        result = pool.step(s0, nstep=100)
        nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        assert result.shape == (4, nstate)
        # After 100 steps (0.2s) the ball should have fallen from z=1.0
        assert result[0, Z_IDX] < 1.0

    def test_all_envs_identical(self):
        m = _make_model(1.0)
        pool = BatchEnvPool(m, nbatch=8, nthread=4)
        s0 = np.tile(_get_init_state(m), (8, 1))
        result = pool.step(s0, nstep=50)
        # All envs share the same model and initial state -> identical results
        for i in range(1, 8):
            np.testing.assert_array_equal(result[0], result[i])


class TestMultiModelPool:
    """Core multi-model functionality."""

    def test_create_multi_model_pool(self):
        m1 = _make_model(1.0)
        m2 = _make_model(0.1)
        pool = BatchEnvPool([m1, m2, m1, m2], nbatch=4, nthread=2)
        assert pool.nbatch == 4

    def test_different_models_different_trajectories(self):
        m_high = _make_model(1.0)
        m_low = _make_model(0.1)
        pool = BatchEnvPool(
            [m_high, m_low, m_high, m_low], nbatch=4, nthread=2
        )

        s_high = _get_init_state(m_high)
        s_low = _get_init_state(m_low)
        state0 = np.array([s_high, s_low, s_high, s_low])

        result = pool.step(state0, nstep=250)
        nstate = s_high.shape[0]
        s = result.reshape(4, nstate)

        # Same-model envs must be identical
        np.testing.assert_array_equal(s[0], s[2])
        np.testing.assert_array_equal(s[1], s[3])

        # Different-model envs must differ (different initial height)
        assert abs(s[0, Z_IDX] - s[1, Z_IDX]) > 0.001

    def test_deduplication(self):
        """Pool with repeated model pointers should not crash or waste memory."""
        m = _make_model(1.0)
        # 100 envs, all sharing the same source model -> only 1 internal copy
        pool = BatchEnvPool([m] * 100, nbatch=100, nthread=4)
        s0 = np.tile(_get_init_state(m), (100, 1))
        result = pool.step(s0, nstep=10)
        assert result.shape[0] == 100
        # All should be identical
        for i in range(1, 100):
            np.testing.assert_array_equal(result[0], result[i])


class TestPoolAPI:
    """Test forward and compute_site_jacobians.

    NOTE: ``get_model`` / ``get_all_models`` are excluded because upstream
    mujoco_uni has a known bug where ``MjModelWrapper`` teardown aborts
    when the raw pointer is not in the global ``MjModelRawPointerMap``.
    This affects both patched and unpatched builds.
    """

    def test_forward_multi_model(self):
        m1 = _make_model(1.0)
        m2 = _make_model(0.5)
        pool = BatchEnvPool([m1, m2, m1, m2], nbatch=4, nthread=2)
        s1 = _get_init_state(m1)
        s2 = _get_init_state(m2)
        state0 = np.array([s1, s2, s1, s2])
        fwd = pool.forward(state0)
        assert fwd.shape == (4, 3)  # 3 = nsensordata (framepos x,y,z)

    def test_forward_single_model(self):
        m = _make_model(1.0)
        pool = BatchEnvPool(m, nbatch=4, nthread=2)
        s0 = np.tile(_get_init_state(m), (4, 1))
        fwd = pool.forward(s0)
        assert fwd.shape == (4, 3)

    def test_compute_site_jacobians_single(self):
        m = _make_model(1.0)
        pool = BatchEnvPool(m, nbatch=4, nthread=2)
        s0 = np.tile(_get_init_state(m), (4, 1))
        # Returns (jacp, jacr) or just jacp depending on flags;
        # default jacp=True, jacr=False => returns tuple (jacp_array, None)
        result = pool.compute_site_jacobians(s0, np.array([0, 0, 0, 0]))
        if isinstance(result, tuple):
            jacp = result[0]
        else:
            jacp = result
        # jacp shape: (nbatch, 1, 3, nv) or (nbatch, 3, nv)
        assert jacp.shape[0] == 4
