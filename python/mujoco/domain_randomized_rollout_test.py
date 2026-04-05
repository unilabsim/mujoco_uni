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
"""Tests for mujoco.domain_randomized_rollout.DomainRandomizedRolloutPool."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import mujoco
from mujoco import domain_randomized_rollout as drr
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


def _rand_initial_state(model, nbatch, rng):
  nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  return rng.standard_normal((nbatch, nstate)).astype(np.float64)


class ConstructorTest(absltest.TestCase):

  def test_nbatch_nthread(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=4, nthread=2
    ) as pool:
      self.assertEqual(pool.nbatch, 4)
      self.assertEqual(pool.nthread, 2)
      self.assertEqual(
          pool.nstate,
          mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS),
      )
      self.assertEqual(pool.nv, model.nv)
      self.assertEqual(pool.nsensordata, model.nsensordata)

  def test_nthread_zero(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    pool = drr.DomainRandomizedRolloutPool(model, nbatch=2, nthread=0)
    try:
      self.assertEqual(pool.nthread, 0)
    finally:
      pool.close()

  def test_invalid_nbatch(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaises(ValueError):
      drr.DomainRandomizedRolloutPool(model, nbatch=0, nthread=1)


class RolloutParityTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)

  @parameterized.parameters((0,), (2,), (4,))
  def test_rollout_parity_vs_stateless(self, nthread):
    nbatch = 8
    nstep = 5
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    state0 = _rand_initial_state(model, nbatch, self.rng)
    control = self.rng.standard_normal(
        (nbatch, nstep, model.nu)
    ).astype(np.float64)

    # Reference path: stateless rollout over a list of identical models.
    # stateless rollout.rollout infers nthread from len(data).
    ref_models = [copy.copy(model) for _ in range(nbatch)]
    ref_datas = (
        [mujoco.MjData(model) for _ in range(nthread)]
        if nthread > 0
        else [mujoco.MjData(model)]
    )
    ref_state, ref_sensordata = stateless_rollout.rollout(
        ref_models,
        ref_datas,
        state0,
        control,
    )

    with drr.DomainRandomizedRolloutPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool:
      state, sensordata = pool.rollout(
          state0,
          nstep=nstep,
          control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
          control=control,
      )

    np.testing.assert_array_equal(state, ref_state)
    np.testing.assert_array_equal(sensordata, ref_sensordata)

  def test_rollout_preallocated_outputs(self):
    nbatch, nstep = 3, 4
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    state0 = _rand_initial_state(model, nbatch, self.rng)
    control = np.zeros((nbatch, nstep, model.nu))
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state = np.zeros((nbatch, nstep, nstate))
    sensordata = np.zeros((nbatch, nstep, model.nsensordata))
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=nbatch, nthread=0
    ) as pool:
      out_state, out_sensor = pool.rollout(
          state0,
          nstep=nstep,
          control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
          control=control,
          state=state,
          sensordata=sensordata,
      )
    # In-place write: identity of the returned array == the preallocated one.
    self.assertIs(out_state, state)
    self.assertIs(out_sensor, sensordata)

  def test_rollout_wrong_state0_shape(self):
    nbatch = 4
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    state0 = _rand_initial_state(model, nbatch + 1, self.rng)
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=nbatch, nthread=0
    ) as pool:
      with self.assertRaises(ValueError):
        pool.rollout(
            state0,
            nstep=2,
            control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
        )


def _reference_reset(model, state0_row, skipsensor=False):
  """Run mj_forward from a given full-physics state on a fresh data."""
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


class ResetSubsetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(7)

  def test_empty_subset_is_noop(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=4, nthread=0
    ) as pool:
      nstate = pool.nstate
      nsensordata = pool.nsensordata
      state, sensor = pool.reset_subset(
          env_ids=np.array([], dtype=np.int32),
          initial_state=np.zeros((0, nstate)),
      )
    self.assertEqual(state.shape, (0, nstate))
    self.assertEqual(sensor.shape, (0, nsensordata))

  @parameterized.parameters((0,), (2,))
  def test_single_env_matches_reference(self, nthread):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nbatch = 4
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool:
      env_ids = np.array([2], dtype=np.int32)
      init = _rand_initial_state(model, 1, self.rng)
      state, sensor = pool.reset_subset(env_ids, init)
      ref_state, ref_sensor = _reference_reset(model, init[0])
    np.testing.assert_allclose(state[0], ref_state, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sensor[0], ref_sensor, atol=1e-12, rtol=0)

  @parameterized.parameters((0,), (4,))
  def test_many_env_matches_reference(self, nthread):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nbatch = 16
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool:
      env_ids = np.array([0, 3, 5, 7, 10, 15], dtype=np.int32)
      init = _rand_initial_state(model, len(env_ids), self.rng)
      state, sensor = pool.reset_subset(env_ids, init)
      for k, _ in enumerate(env_ids):
        ref_state, ref_sensor = _reference_reset(model, init[k])
        np.testing.assert_allclose(state[k], ref_state, atol=1e-12, rtol=0)
        np.testing.assert_allclose(sensor[k], ref_sensor, atol=1e-12, rtol=0)

  def test_env_id_out_of_range(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=4, nthread=0
    ) as pool:
      with self.assertRaises(ValueError):
        pool.reset_subset(
            env_ids=np.array([0, 4], dtype=np.int32),
            initial_state=np.zeros((2, pool.nstate)),
        )

  def test_skipsensor_runs(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nbatch = 2
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=nbatch, nthread=0
    ) as pool:
      env_ids = np.array([0], dtype=np.int32)
      init = _rand_initial_state(model, 1, self.rng)
      state_a, _ = pool.reset_subset(env_ids, init, skipsensor=True)
      state_b, _ = pool.reset_subset(env_ids, init, skipsensor=False)
    # skipsensor should not affect the state output (state is written from
    # mj_getState regardless, and mjSTAGE_NONE with skipsensor only skips
    # sensor evaluation during the forward).
    np.testing.assert_allclose(state_a, state_b, atol=1e-12, rtol=0)


class PatchingAndRefreshTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(11)

  def test_unknown_field_raises(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=2, nthread=0
    ) as pool:
      with self.assertRaises(ValueError):
        pool.reset_subset(
            env_ids=np.array([0], dtype=np.int32),
            initial_state=np.zeros((1, pool.nstate)),
            randomization={"not_a_field": np.zeros((1, 1))},
        )

  def test_wrong_shape_raises(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=2, nthread=0
    ) as pool:
      with self.assertRaises(ValueError):
        pool.reset_subset(
            env_ids=np.array([0], dtype=np.int32),
            initial_state=np.zeros((1, pool.nstate)),
            randomization={"body_mass": np.zeros((1, model.nbody + 1))},
        )

  def test_body_mass_patched_only_for_target(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    nbatch = 4
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=nbatch, nthread=0
    ) as pool:
      original = pool.get_field(0, "body_mass").copy()
      new_mass = np.tile(original, (2, 1)) * 5.0
      env_ids = np.array([1, 3], dtype=np.int32)
      init = _rand_initial_state(model, 2, self.rng)
      pool.reset_subset(
          env_ids, init, randomization={"body_mass": new_mass}
      )
      # Target envs see the new mass; non-target envs untouched.
      np.testing.assert_allclose(pool.get_field(1, "body_mass"), new_mass[0])
      np.testing.assert_allclose(pool.get_field(3, "body_mass"), new_mass[1])
      np.testing.assert_array_equal(pool.get_field(0, "body_mass"), original)
      np.testing.assert_array_equal(pool.get_field(2, "body_mass"), original)

  def test_refresh_field_changes_dynamics(self):
    # body_mass is a refresh field. If mj_setConst is called on touched
    # envs, mj_forward should observe the new inertia. If it's NOT called,
    # dof_M0 etc. stay stale and the output would diverge from a freshly
    # compiled reference model.
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    nbatch = 2
    nstate = mujoco.mj_stateSize(
        model, mujoco.mjtState.mjSTATE_FULLPHYSICS
    )

    # Build a reference model with a 5x mass. Because the reference model
    # goes through mj_compile (via from_xml_string), it has derived
    # constants consistent with the new mass.
    ref_model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    new_mass = ref_model.body_mass.copy() * 5.0
    ref_model.body_mass[:] = new_mass
    # Refresh derived constants on the reference model too.
    ref_data = mujoco.MjData(ref_model)
    mujoco.mj_setConst(ref_model, ref_data)

    init = _rand_initial_state(model, 1, self.rng)
    ref_out_state, _ = _reference_reset(ref_model, init[0])

    with drr.DomainRandomizedRolloutPool(
        model, nbatch=nbatch, nthread=0
    ) as pool:
      env_ids = np.array([0], dtype=np.int32)
      state, _ = pool.reset_subset(
          env_ids,
          init,
          randomization={"body_mass": new_mass.reshape(1, -1)},
      )
    # After mass change + setConst refresh + forward, our qacc should
    # match the reference model.
    np.testing.assert_allclose(state[0], ref_out_state, atol=1e-10, rtol=1e-10)

  def test_geom_friction_write_through(self):
    # Non-refresh field: must update the model tensor but must NOT trigger
    # mj_setConst. Here we verify write-through. (setConst would be a
    # no-op for dof_M0 anyway since mass/armature unchanged, so an
    # "observable refresh" test is not feasible for a non-refresh field;
    # we instead verify the model field is written correctly.)
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=2, nthread=0
    ) as pool:
      original = pool.get_field(0, "geom_friction").copy()
      new_friction = original.reshape(1, -1) * 0.0 + np.array(
          [[1.0, 0.01, 0.0001] * (original.size // 3)]
      )
      env_ids = np.array([0], dtype=np.int32)
      init = _rand_initial_state(model, 1, self.rng)
      pool.reset_subset(
          env_ids, init, randomization={"geom_friction": new_friction}
      )
      np.testing.assert_allclose(
          pool.get_field(0, "geom_friction"), new_friction[0]
      )
      # Env 1 untouched.
      np.testing.assert_array_equal(
          pool.get_field(1, "geom_friction"), original
      )

  def test_sparse_payload_leaves_other_fields_untouched(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with drr.DomainRandomizedRolloutPool(
        model, nbatch=2, nthread=0
    ) as pool:
      orig_armature = pool.get_field(0, "dof_armature").copy()
      env_ids = np.array([0], dtype=np.int32)
      init = _rand_initial_state(model, 1, self.rng)
      new_mass = pool.get_field(0, "body_mass").copy() * 2.0
      pool.reset_subset(
          env_ids,
          init,
          randomization={"body_mass": new_mass.reshape(1, -1)},
      )
      np.testing.assert_array_equal(
          pool.get_field(0, "dof_armature"), orig_armature
      )


if __name__ == "__main__":
  absltest.main()
