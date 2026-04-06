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
"""Tests for mujoco.batch_env.BatchEnvPool."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import mujoco
from mujoco import batch_env
from mujoco import rollout as stateless_rollout


TEST_XML = r"""
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .1">
      <joint name="yaw" axis="0 0 1"/>
      <joint name="pitch" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 1 0 0"/>
      <geom type="box" pos="1 0 0" size=".1 .1 .1"/>
      <site name="site" pos="1 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="pitch" gainprm="100"/>
    <general joint="yaw" dyntype="filter" dynprm="1" gainprm="100"/>
  </actuator>
  <sensor>
    <accelerometer site="site"/>
  </sensor>
</mujoco>
"""

TEST_XML_FREE = r"""
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .5">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def _rand_state(model, nbatch, rng):
  nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  return rng.standard_normal((nbatch, nstate)).astype(np.float64)


def _reference_forward(model, state0_row, skipsensor=False):
  data = mujoco.MjData(model)
  mujoco.mj_setState(
      model, data, state0_row, int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
  )
  if skipsensor:
    mujoco.mj_forwardSkip(model, data, int(mujoco.mjtStage.mjSTAGE_NONE), 1)
  else:
    mujoco.mj_forward(model, data)
  nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  out_state = np.zeros(nstate)
  mujoco.mj_getState(
      model, data, out_state, int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
  )
  return out_state, np.asarray(data.sensordata).copy()


class ConstructorTest(absltest.TestCase):

  def test_basic(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=2) as pool:
      self.assertEqual(pool.nbatch, 4)
      self.assertEqual(pool.nthread, 2)
      self.assertEqual(
          pool.nstate,
          mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS),
      )

  def test_invalid_nbatch(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaises(ValueError):
      batch_env.BatchEnvPool(model, nbatch=0, nthread=1)


class StepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  @parameterized.parameters((0,), (2,), (4,))
  def test_last_state_matches_rollout(self, nthread):
    """step() final state must match stateless rollout's last step."""
    nbatch, nstep = 8, 5
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    state0 = _rand_state(model, nbatch, self.rng)
    control = self.rng.standard_normal(
        (nbatch, nstep, model.nu)
    ).astype(np.float64)

    ref_models = [copy.copy(model) for _ in range(nbatch)]
    ref_datas = (
        [mujoco.MjData(model) for _ in range(nthread)]
        if nthread > 0
        else [mujoco.MjData(model)]
    )
    ref_state, ref_sensor = stateless_rollout.rollout(
        ref_models, ref_datas, state0, control,
    )
    ref_last_state = ref_state[:, -1, :]

    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool:
      state = pool.step(
          state0,
          nstep=nstep,
          control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
          control=control,
      )
      nstate = pool.nstate

    self.assertEqual(state.shape, (nbatch, nstate))
    np.testing.assert_array_equal(state, ref_last_state)

  def test_wrong_shape(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.step(
            _rand_state(model, 5, self.rng),
            nstep=2,
            control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
        )


class ForwardTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(7)

  @parameterized.parameters((0,), (4,))
  def test_matches_reference(self, nthread):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nbatch = 8
    state0 = _rand_state(model, nbatch, self.rng)

    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool:
      sensor = pool.forward(state0)

    for i in range(nbatch):
      _, ref_sensor = _reference_forward(model, state0[i])
      np.testing.assert_allclose(sensor[i], ref_sensor, atol=1e-12, rtol=0)


class ResetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(7)

  def test_empty_subset(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=0) as pool:
      nstate, nsensor = pool.nstate, pool.nsensordata
      state, sensor = pool.reset(
          env_ids=np.array([], dtype=np.int32),
          initial_state=np.zeros((0, nstate)),
      )
    self.assertEqual(state.shape, (0, nstate))
    self.assertEqual(sensor.shape, (0, nsensor))

  @parameterized.parameters((0,), (2,))
  def test_single_env(self, nthread):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(
        model, nbatch=4, nthread=nthread
    ) as pool:
      init = _rand_state(model, 1, self.rng)
      state, sensor = pool.reset(
          env_ids=np.array([2], dtype=np.int32), initial_state=init,
      )
      ref_state, ref_sensor = _reference_forward(model, init[0])
    np.testing.assert_allclose(state[0], ref_state, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sensor[0], ref_sensor, atol=1e-12, rtol=0)

  @parameterized.parameters((0,), (4,))
  def test_many_envs(self, nthread):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nbatch = 16
    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool:
      env_ids = np.array([0, 3, 5, 7, 10, 15], dtype=np.int32)
      init = _rand_state(model, len(env_ids), self.rng)
      state, sensor = pool.reset(env_ids, init)
      for k, _ in enumerate(env_ids):
        ref_state, ref_sensor = _reference_forward(model, init[k])
        np.testing.assert_allclose(state[k], ref_state, atol=1e-12, rtol=0)
        np.testing.assert_allclose(sensor[k], ref_sensor, atol=1e-12, rtol=0)

  def test_env_id_out_of_range(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.reset(
            env_ids=np.array([0, 4], dtype=np.int32),
            initial_state=np.zeros((2, pool.nstate)),
        )


class PatchingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(11)

  def test_unknown_field(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.reset(
            env_ids=np.array([0], dtype=np.int32),
            initial_state=np.zeros((1, pool.nstate)),
            randomization={"not_a_field": np.zeros((1, 1))},
        )

  def test_body_mass_target_only(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=0) as pool:
      original = pool.get_field(0, "body_mass").copy()
      new_mass = np.tile(original, (2, 1)) * 5.0
      env_ids = np.array([1, 3], dtype=np.int32)
      init = _rand_state(model, 2, self.rng)
      pool.reset(env_ids, init, randomization={"body_mass": new_mass})
      np.testing.assert_allclose(pool.get_field(1, "body_mass"), new_mass[0])
      np.testing.assert_allclose(pool.get_field(3, "body_mass"), new_mass[1])
      np.testing.assert_array_equal(pool.get_field(0, "body_mass"), original)
      np.testing.assert_array_equal(pool.get_field(2, "body_mass"), original)

  def test_refresh_field_dynamics(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    ref_model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    new_mass = ref_model.body_mass.copy() * 5.0
    ref_model.body_mass[:] = new_mass
    ref_data = mujoco.MjData(ref_model)
    mujoco.mj_setConst(ref_model, ref_data)

    init = _rand_state(model, 1, self.rng)
    ref_out, _ = _reference_forward(ref_model, init[0])

    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      state, _ = pool.reset(
          env_ids=np.array([0], dtype=np.int32),
          initial_state=init,
          randomization={"body_mass": new_mass.reshape(1, -1)},
      )
    np.testing.assert_allclose(state[0], ref_out, atol=1e-10, rtol=1e-10)

  def test_sparse_payload(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      orig_armature = pool.get_field(0, "dof_armature").copy()
      init = _rand_state(model, 1, self.rng)
      new_mass = pool.get_field(0, "body_mass").copy() * 2.0
      pool.reset(
          np.array([0], dtype=np.int32), init,
          randomization={"body_mass": new_mass.reshape(1, -1)},
      )
      np.testing.assert_array_equal(
          pool.get_field(0, "dof_armature"), orig_armature
      )


if __name__ == "__main__":
  absltest.main()
