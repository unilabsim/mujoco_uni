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

TEST_XML_POSITION = r"""
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .5">
      <joint name="hinge" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 0 0 .3"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="hinge" kp="10" kv="1"/>
  </actuator>
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


def _reference_last_state(model, state0, control):
  data = [mujoco.MjData(model)]
  state_traj, _ = stateless_rollout.rollout([model], data, state0, control)
  return state_traj[:, -1, :]


FIELD_COMPONENT_WIDTHS = {
    "body_mass": 1,
    "body_ipos": 3,
    "body_iquat": 4,
    "body_inertia": 3,
    "dof_armature": 1,
    "geom_friction": 3,
    "kp": 1,
    "kd": 1,
}
REFRESH_FIELDS = (
    "body_mass",
    "body_ipos",
    "body_iquat",
    "body_inertia",
    "dof_armature",
)


def _field_index_count(model, name):
  if name.startswith("body_"):
    return model.nbody
  if name == "dof_armature":
    return model.nv
  if name == "geom_friction":
    return model.ngeom
  if name in ("kp", "kd"):
    return model.nu
  raise ValueError(f"Unsupported field {name}")


def _field_entries(pool, env_id, name, model):
  width = FIELD_COMPONENT_WIDTHS[name]
  flat = pool.get_field(env_id, name).copy()
  if width == 1:
    return flat
  return flat.reshape(_field_index_count(model, name), width)


def _make_replacement_values(name, values):
  arr = np.asarray(values, dtype=np.float64).copy()
  if FIELD_COMPONENT_WIDTHS[name] == 1:
    out = arr + np.linspace(0.25, 0.5, arr.size, dtype=np.float64).reshape(
        arr.shape
    )
    return out.item() if arr.ndim == 0 else out
  if name == "body_iquat":
    presets = np.asarray([
        [0.92387953, 0.0, 0.38268343, 0.0],
        [0.92387953, 0.38268343, 0.0, 0.0],
        [0.96592583, 0.0, 0.0, 0.25881905],
    ])
    out = presets[:arr.shape[0]].copy()
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out if arr.ndim == 2 else out[0]
  scale = 1e-3 if name == "geom_friction" else 1e-2
  return arr + scale * np.arange(1, arr.size + 1, dtype=np.float64).reshape(
      arr.shape
  )


def _set_model_field_indexed(model, name, indices, values):
  indices = np.asarray(indices, dtype=np.int32)
  values = np.asarray(values, dtype=np.float64)
  if name == "body_mass":
    model.body_mass[indices] = values
  elif name == "body_ipos":
    model.body_ipos[indices] = values
  elif name == "body_iquat":
    model.body_iquat[indices] = values
  elif name == "body_inertia":
    model.body_inertia[indices] = values
  elif name == "dof_armature":
    model.dof_armature[indices] = values
  elif name == "geom_friction":
    model.geom_friction[indices] = values
  elif name == "kp":
    for idx, val in zip(indices, values):
      model.actuator_gainprm[idx, 0] = val
      model.actuator_biasprm[idx, 1] = -val
  elif name == "kd":
    for idx, val in zip(indices, values):
      model.actuator_biasprm[idx, 2] = -val
  else:
    raise ValueError(f"Unsupported field {name}")


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

  def test_supported_fields(self):
    self.assertEqual(
        batch_env.SUPPORTED_FIELDS,
        (
            "body_mass",
            "body_ipos",
            "body_iquat",
            "body_inertia",
            "dof_armature",
            "geom_friction",
            "kp",
            "kd",
        ),
    )


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

  def test_removed_dof_damping_field(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.get_field(0, "dof_damping")

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

  def test_body_iquat_target_only(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=0) as pool:
      original = pool.get_field(0, "body_iquat").copy()
      new_iquat = np.tile(original, (2, 1))
      new_iquat[0, 4:8] = np.array([0.92387953, 0.0, 0.38268343, 0.0])
      new_iquat[1, 4:8] = np.array([0.92387953, 0.38268343, 0.0, 0.0])
      env_ids = np.array([1, 3], dtype=np.int32)
      init = _rand_state(model, 2, self.rng)
      pool.reset(env_ids, init, randomization={"body_iquat": new_iquat})
      np.testing.assert_allclose(pool.get_field(1, "body_iquat"), new_iquat[0])
      np.testing.assert_allclose(pool.get_field(3, "body_iquat"), new_iquat[1])
      np.testing.assert_array_equal(pool.get_field(0, "body_iquat"), original)
      np.testing.assert_array_equal(pool.get_field(2, "body_iquat"), original)

  def test_body_inertia_target_only(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=0) as pool:
      original = pool.get_field(0, "body_inertia").copy()
      new_inertia = np.tile(original, (2, 1))
      new_inertia[0, 3:6] *= 2.0
      new_inertia[1, 3:6] *= 3.0
      env_ids = np.array([1, 3], dtype=np.int32)
      init = _rand_state(model, 2, self.rng)
      pool.reset(env_ids, init, randomization={"body_inertia": new_inertia})
      np.testing.assert_allclose(pool.get_field(1, "body_inertia"), new_inertia[0])
      np.testing.assert_allclose(pool.get_field(3, "body_inertia"), new_inertia[1])
      np.testing.assert_array_equal(pool.get_field(0, "body_inertia"), original)
      np.testing.assert_array_equal(pool.get_field(2, "body_inertia"), original)

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

  def test_kp_kd_step_matches_reference_and_target_only(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_POSITION)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state0 = np.zeros((2, nstate), dtype=np.float64)
    control = np.full((2, 1, model.nu), 0.5, dtype=np.float64)
    kp = np.array([[25.0]], dtype=np.float64)
    kd = np.array([[3.0]], dtype=np.float64)

    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      pool.reset(
          env_ids=np.array([0], dtype=np.int32),
          initial_state=state0[:1],
          randomization={"kp": kp, "kd": kd},
      )

      np.testing.assert_allclose(pool.get_field(0, "kp"), kp[0])
      np.testing.assert_allclose(pool.get_field(0, "kd"), kd[0])
      np.testing.assert_allclose(pool.get_field(1, "kp"), np.array([10.0]))
      np.testing.assert_allclose(pool.get_field(1, "kd"), np.array([1.0]))

      state = pool.step(
          state0,
          nstep=1,
          control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
          control=control,
      )

    ref_patched = mujoco.MjModel.from_xml_string(TEST_XML_POSITION)
    ref_patched.actuator_gainprm[:, 0] = kp[0]
    ref_patched.actuator_biasprm[:, 1] = -kp[0]
    ref_patched.actuator_biasprm[:, 2] = -kd[0]
    ref_base = mujoco.MjModel.from_xml_string(TEST_XML_POSITION)

    ref_state_patched = _reference_last_state(ref_patched, state0[:1], control[:1])
    ref_state_base = _reference_last_state(ref_base, state0[:1], control[:1])

    np.testing.assert_allclose(state[0], ref_state_patched[0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(state[1], ref_state_base[0], atol=1e-12, rtol=0)


class IndexedFieldTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(19)

  @parameterized.parameters(*batch_env.SUPPORTED_FIELDS)
  def test_get_field_indexed_single(self, field_name):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    index = min(1, _field_index_count(model, field_name) - 1)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      expected = _field_entries(pool, 0, field_name, model)[index]
      got = pool.get_field_indexed(0, field_name, index)
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)

  @parameterized.parameters(*batch_env.SUPPORTED_FIELDS)
  def test_get_field_indexed_many(self, field_name):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    index_count = _field_index_count(model, field_name)
    indices = np.array([index_count - 1, 0], dtype=np.int32)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      expected = _field_entries(pool, 0, field_name, model)[indices]
      got = pool.get_field_indexed(0, field_name, indices)
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)

  @parameterized.parameters(*batch_env.SUPPORTED_FIELDS)
  def test_set_field_indexed_single_target_only(self, field_name):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    index = min(1, _field_index_count(model, field_name) - 1)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      original_other = pool.get_field(0, field_name).copy()
      original_target = _field_entries(pool, 1, field_name, model).copy()
      replacement = _make_replacement_values(field_name, original_target[index])
      pool.set_field_indexed(1, field_name, index, replacement)
      updated_target = _field_entries(pool, 1, field_name, model)
      np.testing.assert_array_equal(pool.get_field(0, field_name), original_other)
      np.testing.assert_allclose(updated_target[index], replacement, atol=1e-12, rtol=0)
      keep_mask = np.ones(_field_index_count(model, field_name), dtype=bool)
      keep_mask[index] = False
      np.testing.assert_allclose(
          updated_target[keep_mask], original_target[keep_mask], atol=1e-12, rtol=0
      )

  @parameterized.parameters(*batch_env.SUPPORTED_FIELDS)
  def test_set_field_indexed_many_target_only(self, field_name):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    indices = np.array([0, _field_index_count(model, field_name) - 1], dtype=np.int32)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      original_other = pool.get_field(0, field_name).copy()
      original_target = _field_entries(pool, 1, field_name, model).copy()
      replacement = _make_replacement_values(field_name, original_target[indices])
      pool.set_field_indexed(1, field_name, indices, replacement)
      updated_target = _field_entries(pool, 1, field_name, model)
      np.testing.assert_array_equal(pool.get_field(0, field_name), original_other)
      np.testing.assert_allclose(
          updated_target[indices], replacement, atol=1e-12, rtol=0
      )
      keep_mask = np.ones(_field_index_count(model, field_name), dtype=bool)
      keep_mask[indices] = False
      np.testing.assert_allclose(
          updated_target[keep_mask], original_target[keep_mask], atol=1e-12, rtol=0
      )

  @parameterized.parameters(*batch_env.SUPPORTED_FIELDS)
  def test_set_field_indexed_repeated_indices_last_wins(self, field_name):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    indices = np.array([0, 0], dtype=np.int32)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      original = _field_entries(pool, 1, field_name, model).copy()
      replacement = _make_replacement_values(field_name, original[indices])
      pool.set_field_indexed(1, field_name, indices, replacement)
      updated = _field_entries(pool, 1, field_name, model)
      np.testing.assert_allclose(updated[0], replacement[-1], atol=1e-12, rtol=0)
      np.testing.assert_allclose(updated[1:], original[1:], atol=1e-12, rtol=0)

  @parameterized.parameters(*batch_env.SUPPORTED_FIELDS)
  def test_get_field_indexed_out_of_range_raises(self, field_name):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.get_field_indexed(0, field_name, _field_index_count(model, field_name))

  @parameterized.parameters(*batch_env.SUPPORTED_FIELDS)
  def test_set_field_indexed_out_of_range_raises(self, field_name):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    width = FIELD_COMPONENT_WIDTHS[field_name]
    bad_index = _field_index_count(model, field_name)
    replacement = 1.0 if width == 1 else np.ones((width,), dtype=np.float64)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.set_field_indexed(0, field_name, bad_index, replacement)

  @parameterized.parameters(*batch_env.SUPPORTED_FIELDS)
  def test_set_field_indexed_single_wrong_shape_raises(self, field_name):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    width = FIELD_COMPONENT_WIDTHS[field_name]
    bad_value = (
        np.ones((2,), dtype=np.float64)
        if width == 1
        else np.ones((2, width), dtype=np.float64)
    )
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.set_field_indexed(0, field_name, 0, bad_value)

  @parameterized.parameters(*batch_env.SUPPORTED_FIELDS)
  def test_set_field_indexed_many_wrong_shape_raises(self, field_name):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    width = FIELD_COMPONENT_WIDTHS[field_name]
    bad_value = (
        np.ones((3,), dtype=np.float64)
        if width == 1
        else np.ones((3, width), dtype=np.float64)
    )
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.set_field_indexed(0, field_name, [0, 1], bad_value)

  @parameterized.parameters(*REFRESH_FIELDS)
  def test_set_field_indexed_refresh_fields_match_reference_dynamics(
      self, field_name
  ):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    ref_model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    if field_name == "dof_armature":
      indices = np.array([0, 2], dtype=np.int32)
    else:
      indices = np.array([1], dtype=np.int32)

    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      original = _field_entries(pool, 0, field_name, model)
      replacement = _make_replacement_values(field_name, original[indices])
      pool.set_field_indexed(0, field_name, indices, replacement)
      _set_model_field_indexed(ref_model, field_name, indices, replacement)
      ref_data = mujoco.MjData(ref_model)
      mujoco.mj_setConst(ref_model, ref_data)

      init = _rand_state(model, 1, self.rng)
      state, _ = pool.reset(
          env_ids=np.array([0], dtype=np.int32),
          initial_state=init,
      )
      ref_state, _ = _reference_forward(ref_model, init[0])

    np.testing.assert_allclose(state[0], ref_state, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
  absltest.main()
