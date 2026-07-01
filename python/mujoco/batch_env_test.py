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

TEST_XML_TWO_SITES = r"""
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .1">
      <joint name="yaw" axis="0 0 1"/>
      <joint name="pitch" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 1 0 0"/>
      <site name="tip" pos="1 0 0"/>
      <body pos=".5 0 0">
        <joint name="elbow" axis="0 1 0"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .4 0 0"/>
        <site name="hand" pos=".4 0 0"/>
      </body>
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

TEST_XML_HFIELD = r"""
<mujoco>
  <asset>
    <hfield name="terrain" nrow="2" ncol="2" size="1 1 1 .1"
            elevation="0 1
                       0 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 0"/>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom name="base_geom" type="sphere" size=".05"/>
    </body>
  </worldbody>
</mujoco>
"""

TEST_XML_RAY = r"""
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1"/>
    <geom name="wall" type="box" pos="1 0 0.5" size=".1 .5 .5"/>
    <body name="caster" pos="0 0 .5">
      <freejoint/>
      <geom name="body_geom" type="sphere" size=".05"/>
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


def _reference_last_state(model, state0, control):
  data = [mujoco.MjData(model)]
  state_traj, _ = stateless_rollout.rollout([model], data, state0, control)
  return state_traj[:, -1, :]


def _reference_multi_ray(
    model,
    state0_row,
    pnt,
    vec,
    *,
    geomgroup=None,
    flg_static=1,
    bodyexclude=-1,
    return_normal=False,
    cutoff=None,
):
  data = mujoco.MjData(model)
  mujoco.mj_setState(
      model, data, state0_row, int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
  )
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)

  vec = np.asarray(vec, dtype=np.float64)
  nray = vec.shape[0]
  geomid = np.zeros(nray, dtype=np.int32)
  dist = np.zeros(nray, dtype=np.float64)
  normal = np.zeros((nray, 3), dtype=np.float64) if return_normal else None
  mujoco.mj_multiRay(
      m=model,
      d=data,
      pnt=np.asarray(pnt, dtype=np.float64),
      vec=vec.reshape(-1),
      geomgroup=geomgroup,
      flg_static=flg_static,
      bodyexclude=bodyexclude,
      geomid=geomid,
      dist=dist,
      normal=None if normal is None else normal.reshape(-1),
      nray=nray,
      cutoff=mujoco.mjMAXVAL if cutoff is None else cutoff,
  )
  return dist, geomid, normal


def _hfield_model(height_data):
  model = mujoco.MjModel.from_xml_string(TEST_XML_HFIELD)
  np.asarray(model.hfield_data)[:] = np.asarray(
      height_data, dtype=np.float32
  ).reshape(-1)
  return model


def _state_from_free_qpos(model, qpos_rows):
  qpos_rows = np.asarray(qpos_rows, dtype=np.float64)
  if qpos_rows.ndim == 1:
    qpos_rows = qpos_rows[None, :]
  nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  state = np.zeros((qpos_rows.shape[0], nstate), dtype=np.float64)
  for i, qpos in enumerate(qpos_rows):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    mujoco.mj_getState(
        model, data, state[i], int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    )
  return state


def _free_qpos(x=0.0, y=0.0, z=1.5, yaw=0.0):
  return np.array(
      [x, y, z, np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
      dtype=np.float64,
  )


FIELD_COMPONENT_WIDTHS = {
    "body_mass": 1,
    "body_ipos": 3,
    "body_iquat": 4,
    "body_inertia": 3,
    "dof_armature": 1,
    "gravity": 3,
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
  if name == "gravity":
    return 1
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
  elif name == "gravity":
    for idx, val in zip(indices, values.reshape(-1, 3)):
      if idx != 0:
        raise ValueError("gravity only supports index 0")
      model.opt.gravity[:] = val
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

  def test_model_sequence_copies_per_env(self):
    model0 = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    model1 = copy.copy(model0)
    model1.body_mass[:] *= 2.0
    model1.geom_friction[1] = np.array([0.9, 0.01, 0.001], dtype=np.float64)

    with batch_env.BatchEnvPool([model0, model1], nbatch=2, nthread=0) as pool:
      np.testing.assert_allclose(
          pool.get_field(0, "body_mass"), np.asarray(model0.body_mass).reshape(-1)
      )
      np.testing.assert_allclose(
          pool.get_field(1, "body_mass"), np.asarray(model1.body_mass).reshape(-1)
      )
      np.testing.assert_allclose(
          pool.get_field(0, "geom_friction"),
          np.asarray(model0.geom_friction).reshape(-1),
      )
      np.testing.assert_allclose(
          pool.get_field(1, "geom_friction"),
          np.asarray(model1.geom_friction).reshape(-1),
      )

  def test_model_sequence_length_one_expands(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with batch_env.BatchEnvPool([model], nbatch=3, nthread=0) as pool:
      expected = np.asarray(model.body_mass).reshape(-1)
      for env_id in range(3):
        np.testing.assert_allclose(pool.get_field(env_id, "body_mass"), expected)

  def test_invalid_model_sequence_length(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaises(ValueError):
      batch_env.BatchEnvPool([model, copy.copy(model)], nbatch=3, nthread=0)

  def test_incompatible_model_sequence(self):
    model0 = mujoco.MjModel.from_xml_string(TEST_XML)
    model1 = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with self.assertRaises(ValueError):
      batch_env.BatchEnvPool([model0, model1], nbatch=2, nthread=0)

  def test_get_model_returns_pool_owned_reference(self):
    model0 = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    model1 = copy.copy(model0)
    model1.body_mass[:] *= 2.0

    with batch_env.BatchEnvPool([model0, model1], nbatch=2, nthread=0) as pool:
      pooled_model = pool.get_model(1)
      self.assertIsInstance(pooled_model, mujoco.MjModel)
      self.assertEqual(pooled_model._address, pool.get_model(1)._address)

      pooled_model.body_mass[:] *= 3.0
      np.testing.assert_allclose(
          pool.get_field(1, "body_mass"),
          np.asarray(pooled_model.body_mass).reshape(-1),
      )
      np.testing.assert_allclose(
          pool.get_field(0, "body_mass"),
          np.asarray(model0.body_mass).reshape(-1),
      )

  def test_get_model_many_returns_selected_references(self):
    model0 = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    model1 = copy.copy(model0)
    model2 = copy.copy(model0)
    model1.body_mass[:] *= 2.0
    model2.body_mass[:] *= 4.0

    with batch_env.BatchEnvPool(
        [model0, model1, model2], nbatch=3, nthread=0
    ) as pool:
      models = pool.get_model([2, 0])
      self.assertLen(models, 2)
      self.assertEqual(models[0]._address, pool.get_model(2)._address)
      self.assertEqual(models[1]._address, pool.get_model(0)._address)
      np.testing.assert_allclose(
          np.asarray(models[0].body_mass).reshape(-1),
          pool.get_field(2, "body_mass"),
      )
      np.testing.assert_allclose(
          np.asarray(models[1].body_mass).reshape(-1),
          pool.get_field(0, "body_mass"),
      )

  def test_get_all_models_returns_all_references(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)

    with batch_env.BatchEnvPool(model, nbatch=3, nthread=0) as pool:
      models = pool.get_all_models()
      self.assertLen(models, 3)

      models[1].body_mass[:] *= 5.0
      np.testing.assert_allclose(
          pool.get_field(1, "body_mass"),
          np.asarray(models[1].body_mass).reshape(-1),
      )
  def test_supported_fields(self):
    self.assertEqual(
        batch_env.SUPPORTED_FIELDS,
        (
            "body_mass",
            "body_ipos",
            "body_iquat",
            "body_inertia",
            "dof_armature",
            "gravity",
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

  @parameterized.parameters((0,), (4,))
  def test_last_state_and_sensor_matches_rollout(self, nthread):
    """step(return_sensor=True) returns only rollout's final state/sensor."""
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

    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool:
      state, sensor = pool.step(
          state0,
          nstep=nstep,
          control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
          control=control,
          return_sensor=True,
      )
      nstate = pool.nstate

    self.assertEqual(state.shape, (nbatch, nstate))
    self.assertEqual(sensor.shape, (nbatch, model.nsensordata))
    np.testing.assert_array_equal(state, ref_state[:, -1, :])
    np.testing.assert_allclose(sensor, ref_sensor[:, -1, :], atol=1e-12, rtol=0)

  @parameterized.parameters((0,), (4,))
  def test_post_step_forward_sensor_matches_forward_refresh(self, nthread):
    """post_step_forward_sensor matches the previous step()+forward() path."""
    nbatch, nstep = 8, 3
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    state0 = _rand_state(model, nbatch, self.rng)
    control = self.rng.standard_normal(
        (nbatch, nstep, model.nu)
    ).astype(np.float64)

    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool:
      state, sensor = pool.step(
          state0,
          nstep=nstep,
          control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
          control=control,
          return_sensor=True,
          post_step_forward_sensor=True,
      )
      refreshed_sensor = pool.forward(state)

    np.testing.assert_allclose(sensor, refreshed_sensor, atol=1e-12, rtol=0)

  def test_wrong_shape(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.step(
            _rand_state(model, 5, self.rng),
            nstep=2,
            control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
        )

  def test_distinct_models_match_per_env_references(self):
    model0 = mujoco.MjModel.from_xml_string(TEST_XML_POSITION)
    model1 = copy.copy(model0)
    model1.actuator_gainprm[:, 0] = 25.0
    model1.actuator_biasprm[:, 1] = -25.0
    model1.actuator_biasprm[:, 2] = -3.0

    nstate = mujoco.mj_stateSize(model0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state0 = np.zeros((2, nstate), dtype=np.float64)
    control = np.full((2, 1, model0.nu), 0.5, dtype=np.float64)

    with batch_env.BatchEnvPool([model0, model1], nbatch=2, nthread=0) as pool:
      state = pool.step(
          state0,
          nstep=1,
          control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
          control=control,
      )

    ref0 = _reference_last_state(model0, state0[:1], control[:1])[0]
    ref1 = _reference_last_state(model1, state0[:1], control[1:])[0]
    np.testing.assert_allclose(state[0], ref0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(state[1], ref1, atol=1e-12, rtol=0)


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

  def test_distinct_models_match_per_env_references(self):
    model0 = mujoco.MjModel.from_xml_string(TEST_XML)
    model1 = copy.copy(model0)
    model1.body_mass[:] *= 1.5
    model1.body_ipos[1] += np.array([0.01, -0.02, 0.03], dtype=np.float64)
    ref_data = mujoco.MjData(model1)
    mujoco.mj_setConst(model1, ref_data)

    state0 = _rand_state(model0, 2, self.rng)

    with batch_env.BatchEnvPool([model0, model1], nbatch=2, nthread=0) as pool:
      sensor = pool.forward(state0)

    _, ref_sensor0 = _reference_forward(model0, state0[0])
    _, ref_sensor1 = _reference_forward(model1, state0[1])
    np.testing.assert_allclose(sensor[0], ref_sensor0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sensor[1], ref_sensor1, atol=1e-12, rtol=0)


class HfieldSamplerTest(parameterized.TestCase):

  @parameterized.parameters((0,), (2,))
  def test_height_clearance_and_clamp(self, nthread):
    model = _hfield_model([[0.0, 0.2], [0.4, 1.0]])
    hfield_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"
    )
    frame_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base"
    )
    state0 = np.repeat(
        _state_from_free_qpos(model, _free_qpos(z=1.5)), 2, axis=0
    )
    offsets = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    with batch_env.BatchEnvPool(model, nbatch=2, nthread=nthread) as pool:
      height = pool.sample_hfield_height(
          state0, hfield_geom_id, offsets, frame_body_id, alignment="world"
      )
      clearance = pool.sample_hfield_height(
          state0,
          hfield_geom_id,
          offsets,
          frame_body_id,
          alignment="world",
          output="clearance",
      )

    expected_height = np.array([0.4, 1.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(
        height, np.repeat(expected_height[None, :], 2, axis=0), atol=1e-12
    )
    np.testing.assert_allclose(
        clearance,
        np.repeat((1.5 - expected_height)[None, :], 2, axis=0),
        atol=1e-12,
    )

  def test_yaw_alignment_rotates_offsets(self):
    model = _hfield_model([[0.0, 1.0], [0.0, 1.0]])
    hfield_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"
    )
    frame_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base"
    )
    state0 = _state_from_free_qpos(model, _free_qpos(yaw=np.pi / 2))
    offsets = np.array([[1.0, 0.0]])

    with batch_env.BatchEnvPool(model, nbatch=1, nthread=0) as pool:
      yaw_height = pool.sample_hfield_height(
          state0, hfield_geom_id, offsets, frame_body_id, alignment="yaw"
      )
      world_height = pool.sample_hfield_height(
          state0, hfield_geom_id, offsets, frame_body_id, alignment="world"
      )

    np.testing.assert_allclose(yaw_height, [[0.5]], atol=1e-12)
    np.testing.assert_allclose(world_height, [[1.0]], atol=1e-12)

  def test_model_variants_use_per_env_hfield_data(self):
    model0 = _hfield_model([[0.0, 0.0], [0.0, 0.0]])
    model1 = _hfield_model([[1.0, 1.0], [1.0, 1.0]])
    hfield_geom_id = mujoco.mj_name2id(
        model0, mujoco.mjtObj.mjOBJ_GEOM, "terrain"
    )
    frame_body_id = mujoco.mj_name2id(
        model0, mujoco.mjtObj.mjOBJ_BODY, "base"
    )
    state0 = np.repeat(
        _state_from_free_qpos(model0, _free_qpos()), 2, axis=0
    )

    with batch_env.BatchEnvPool([model0, model1], nbatch=2, nthread=2) as pool:
      height = pool.sample_hfield_height(
          state0, hfield_geom_id, [[0.0, 0.0]], frame_body_id
      )

    np.testing.assert_allclose(height, [[0.0], [1.0]], atol=1e-12)

  def test_validation_errors(self):
    model = _hfield_model([[0.0, 1.0], [0.0, 1.0]])
    hfield_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"
    )
    frame_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base"
    )
    non_hfield_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "base_geom"
    )
    state0 = _state_from_free_qpos(model, _free_qpos())

    with batch_env.BatchEnvPool(model, nbatch=1, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.sample_hfield_height(
            state0, hfield_geom_id, np.zeros((2, 3)), frame_body_id
        )
      with self.assertRaises(ValueError):
        pool.sample_hfield_height(
            state0, non_hfield_geom_id, [[0.0, 0.0]], frame_body_id
        )
      with self.assertRaises(ValueError):
        pool.sample_hfield_height(
            state0,
            hfield_geom_id,
            [[0.0, 0.0]],
            frame_body_id,
            output="distance",
        )
      with self.assertRaises(ValueError):
        pool.sample_hfield_height(
            state0,
            hfield_geom_id,
            [[0.0, 0.0]],
            frame_body_id,
            chunk_size=0,
        )


class MultiRayTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(123)

  @parameterized.parameters((0,), (2,))
  def test_matches_reference_shared_origin_and_rays(self, nthread):
    model = mujoco.MjModel.from_xml_string(TEST_XML_RAY)
    state0 = _state_from_free_qpos(
        model,
        np.stack([
            _free_qpos(x=0.0, y=0.0, z=0.5),
            _free_qpos(x=0.2, y=0.0, z=0.5),
            _free_qpos(x=-0.1, y=0.1, z=0.5),
        ]),
    )
    pnt = np.array([0.0, 0.0, 0.6], dtype=np.float64)
    vec = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )

    with batch_env.BatchEnvPool(model, nbatch=3, nthread=nthread) as pool:
      dist, geomid, normal = pool.multi_ray(
          state0, pnt, vec, return_normal=True
      )

    self.assertEqual(dist.shape, (3, 3))
    self.assertEqual(geomid.shape, (3, 3))
    self.assertEqual(normal.shape, (3, 3, 3))
    for i in range(3):
      ref_dist, ref_geomid, ref_normal = _reference_multi_ray(
          model, state0[i], pnt, vec, return_normal=True
      )
      np.testing.assert_allclose(dist[i], ref_dist, atol=1e-12, rtol=0)
      np.testing.assert_array_equal(geomid[i], ref_geomid)
      np.testing.assert_allclose(normal[i], ref_normal, atol=1e-12, rtol=0)

  def test_matches_reference_batched_origin_and_bodyexclude(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_RAY)
    caster_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "caster"
    )
    state0 = _state_from_free_qpos(
        model,
        np.stack([
            _free_qpos(x=0.0, y=0.0, z=0.5),
            _free_qpos(x=0.3, y=0.0, z=0.5),
        ]),
    )
    pnt = np.array([[0.0, 0.0, 0.55], [0.3, 0.0, 0.55]], dtype=np.float64)
    vec = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    bodyexclude = np.array([caster_body_id, -1], dtype=np.int32)

    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      dist, geomid, normal = pool.multi_ray(
          state0,
          pnt,
          vec,
          bodyexclude=bodyexclude,
          return_normal=False,
      )

    self.assertIsNone(normal)
    for i in range(2):
      ref_dist, ref_geomid, _ = _reference_multi_ray(
          model,
          state0[i],
          pnt[i],
          vec,
          bodyexclude=int(bodyexclude[i]),
          return_normal=False,
      )
      np.testing.assert_allclose(dist[i], ref_dist, atol=1e-12, rtol=0)
      np.testing.assert_array_equal(geomid[i], ref_geomid)

  def test_matches_reference_multi_model_batched_vec(self):
    model0 = mujoco.MjModel.from_xml_string(TEST_XML_RAY)
    model1 = copy.copy(model0)
    model1.geom_pos[1] = np.array([1.4, 0.0, 0.5], dtype=np.float64)
    ref_data = mujoco.MjData(model1)
    mujoco.mj_setConst(model1, ref_data)

    state0 = _state_from_free_qpos(
        model0,
        np.stack([
            _free_qpos(x=0.0, y=0.0, z=0.5),
            _free_qpos(x=0.0, y=0.0, z=0.5),
        ]),
    )
    pnt = np.array([0.0, 0.0, 0.6], dtype=np.float64)
    vec = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        ],
        dtype=np.float64,
    )

    with batch_env.BatchEnvPool([model0, model1], nbatch=2, nthread=2) as pool:
      dist, geomid, _ = pool.multi_ray(state0, pnt, vec)

    ref_dist0, ref_geomid0, _ = _reference_multi_ray(
        model0, state0[0], pnt, vec[0]
    )
    ref_dist1, ref_geomid1, _ = _reference_multi_ray(
        model1, state0[1], pnt, vec[1]
    )
    np.testing.assert_allclose(dist[0], ref_dist0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(dist[1], ref_dist1, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(geomid[0], ref_geomid0)
    np.testing.assert_array_equal(geomid[1], ref_geomid1)

  def test_validation_errors(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_RAY)
    state0 = _state_from_free_qpos(model, _free_qpos())
    vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    with batch_env.BatchEnvPool(model, nbatch=1, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.multi_ray(np.zeros((1, pool.nstate + 1)), [0, 0, 0], vec)
      with self.assertRaises(ValueError):
        pool.multi_ray(state0, np.zeros((1, 2)), vec)
      with self.assertRaises(ValueError):
        pool.multi_ray(state0, [0, 0, 0], np.zeros((1, 2)))
      with self.assertRaises(ValueError):
        pool.multi_ray(
            state0, [0, 0, 0], vec, bodyexclude=np.array([0, 1], dtype=np.int32)
        )
      with self.assertRaises(ValueError):
        pool.multi_ray(
            state0, [0, 0, 0], vec, geomgroup=np.zeros((2,), dtype=np.uint8)
        )
      with self.assertRaises(TypeError):
        pool.multi_ray(state0, [0, 0, 0], vec, bodyexclude=np.array([0.5]))


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

  def test_distinct_models_match_per_env_references(self):
    model0 = mujoco.MjModel.from_xml_string(TEST_XML)
    model1 = copy.copy(model0)
    model1.body_mass[:] *= 1.25
    model1.body_iquat[4:8] = np.array([0.92387953, 0.0, 0.38268343, 0.0])
    ref_data = mujoco.MjData(model1)
    mujoco.mj_setConst(model1, ref_data)

    env_ids = np.array([0, 1], dtype=np.int32)
    init = _rand_state(model0, 2, self.rng)

    with batch_env.BatchEnvPool([model0, model1], nbatch=2, nthread=0) as pool:
      state, sensor = pool.reset(env_ids, init)

    ref_state0, ref_sensor0 = _reference_forward(model0, init[0])
    ref_state1, ref_sensor1 = _reference_forward(model1, init[1])
    np.testing.assert_allclose(state[0], ref_state0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(state[1], ref_state1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sensor[0], ref_sensor0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sensor[1], ref_sensor1, atol=1e-12, rtol=0)


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

  def test_gravity_target_only(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML_FREE)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=0) as pool:
      original = pool.get_field(0, "gravity").copy()
      new_gravity = np.array(
          [
              [0.0, -3.0, -6.0],
              [4.0, 0.0, -12.0],
          ],
          dtype=np.float64,
      )
      env_ids = np.array([1, 3], dtype=np.int32)
      init = _rand_state(model, 2, self.rng)
      pool.reset(env_ids, init, randomization={"gravity": new_gravity})
      np.testing.assert_allclose(pool.get_field(1, "gravity"), new_gravity[0])
      np.testing.assert_allclose(pool.get_field(3, "gravity"), new_gravity[1])
      np.testing.assert_array_equal(pool.get_field(0, "gravity"), original)
      np.testing.assert_array_equal(pool.get_field(2, "gravity"), original)

  def test_gravity_reset_matches_reference_without_setconst(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    ref_model = mujoco.MjModel.from_xml_string(TEST_XML)
    new_gravity = np.array([0.5, -1.0, -6.25], dtype=np.float64)
    ref_model.opt.gravity[:] = new_gravity

    init = _rand_state(model, 1, self.rng)
    ref_out, ref_sensor = _reference_forward(ref_model, init[0])

    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      state, sensor = pool.reset(
          env_ids=np.array([0], dtype=np.int32),
          initial_state=init,
          randomization={"gravity": new_gravity.reshape(1, -1)},
      )
    np.testing.assert_allclose(state[0], ref_out, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sensor[0], ref_sensor, atol=1e-12, rtol=0)

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
    index_count = _field_index_count(model, field_name)
    indices = (
        np.array([0], dtype=np.int32)
        if index_count == 1
        else np.array([0, index_count - 1], dtype=np.int32)
    )
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


def _reference_site_jacobian(model, state0_row, site_id, jacp, jacr):
  data = mujoco.MjData(model)
  mujoco.mj_setState(
      model, data, state0_row, int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
  )
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  jp = np.zeros((3, model.nv), dtype=np.float64) if jacp else None
  jr = np.zeros((3, model.nv), dtype=np.float64) if jacr else None
  mujoco.mj_jacSite(model, data, jp, jr, site_id)
  return jp, jr


class JacobianTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(11)

  @parameterized.parameters((0, True, False), (0, False, True), (0, True, True),
                            (4, True, True))
  def test_matches_reference_single_model(self, nthread, jacp, jacr):
    model = mujoco.MjModel.from_xml_string(TEST_XML_TWO_SITES)
    nbatch = 6
    state0 = _rand_state(model, nbatch, self.rng)
    tip_id = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_SITE), "tip"
    )
    hand_id = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_SITE), "hand"
    )
    site_ids = [tip_id, hand_id]

    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, nthread=nthread
    ) as pool:
      jp, jr = pool.compute_site_jacobians(
          state0, site_ids, jacp=jacp, jacr=jacr
      )

    if jacp:
      self.assertEqual(jp.shape, (nbatch, len(site_ids), 3, model.nv))
    else:
      self.assertIsNone(jp)
    if jacr:
      self.assertEqual(jr.shape, (nbatch, len(site_ids), 3, model.nv))
    else:
      self.assertIsNone(jr)

    for i in range(nbatch):
      for k, site in enumerate(site_ids):
        ref_jp, ref_jr = _reference_site_jacobian(
            model, state0[i], site, jacp, jacr
        )
        if jacp:
          np.testing.assert_allclose(jp[i, k], ref_jp, atol=1e-12, rtol=0)
        if jacr:
          np.testing.assert_allclose(jr[i, k], ref_jr, atol=1e-12, rtol=0)

  def test_scalar_site_squeezes_K_dim(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nbatch = 4
    state0 = _rand_state(model, nbatch, self.rng)
    site_id = mujoco.mj_name2id(
        model, int(mujoco.mjtObj.mjOBJ_SITE), "site"
    )

    with batch_env.BatchEnvPool(model, nbatch=nbatch, nthread=0) as pool:
      jp, jr = pool.compute_site_jacobians(
          state0, site_id, jacp=True, jacr=True
      )

    self.assertEqual(jp.shape, (nbatch, 3, model.nv))
    self.assertEqual(jr.shape, (nbatch, 3, model.nv))
    for i in range(nbatch):
      ref_jp, ref_jr = _reference_site_jacobian(
          model, state0[i], site_id, True, True
      )
      np.testing.assert_allclose(jp[i], ref_jp, atol=1e-12, rtol=0)
      np.testing.assert_allclose(jr[i], ref_jr, atol=1e-12, rtol=0)

  def test_matches_reference_multi_model(self):
    model0 = mujoco.MjModel.from_xml_string(TEST_XML_TWO_SITES)
    model1 = copy.copy(model0)
    model1.body_mass[:] *= 1.7
    model1.body_ipos[1] += np.array([0.01, -0.02, 0.03], dtype=np.float64)
    ref_data = mujoco.MjData(model1)
    mujoco.mj_setConst(model1, ref_data)

    state0 = _rand_state(model0, 2, self.rng)
    tip_id = mujoco.mj_name2id(
        model0, int(mujoco.mjtObj.mjOBJ_SITE), "tip"
    )

    with batch_env.BatchEnvPool([model0, model1], nbatch=2, nthread=2) as pool:
      jp, _ = pool.compute_site_jacobians(state0, [tip_id], jacp=True)

    ref_jp0, _ = _reference_site_jacobian(model0, state0[0], tip_id, True, False)
    ref_jp1, _ = _reference_site_jacobian(model1, state0[1], tip_id, True, False)
    np.testing.assert_allclose(jp[0, 0], ref_jp0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(jp[1, 0], ref_jp1, atol=1e-12, rtol=0)

  def test_requires_at_least_one_flag(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    state0 = _rand_state(model, 2, self.rng)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.compute_site_jacobians(state0, [0], jacp=False, jacr=False)

  def test_invalid_site_id(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    state0 = _rand_state(model, 2, self.rng)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.compute_site_jacobians(state0, [model.nsite], jacp=True)

  def test_wrong_state_shape(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    bad = _rand_state(model, 3, self.rng)
    with batch_env.BatchEnvPool(model, nbatch=4, nthread=0) as pool:
      with self.assertRaises(ValueError):
        pool.compute_site_jacobians(bad, [0], jacp=True)


import os as _os
import sys as _sys


def _linux_allowed_cpus():
  """Return the CPU ids this process is allowed to run on, on Linux."""
  if _sys.platform != "linux":
    return []
  return sorted(_os.sched_getaffinity(0))


class NumaPolicyTest(absltest.TestCase):
  """Phase 1 NUMA policy plumbing: numa_policy / cpu_ids / first_touch."""

  def test_default_policy_is_off(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=2) as pool:
      self.assertEqual(pool.numa_policy, "off")
      self.assertEqual(pool.cpu_ids, ())
      self.assertTrue(pool.first_touch)

  def test_pin_requires_cpu_ids(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaises(ValueError):
      batch_env.BatchEnvPool(model, nbatch=2, nthread=2, numa_policy="pin")

  def test_pin_requires_matching_length(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaises(ValueError):
      batch_env.BatchEnvPool(
          model, nbatch=2, nthread=2, numa_policy="pin", cpu_ids=[0]
      )

  def test_pin_requires_positive_nthread(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaises(ValueError):
      batch_env.BatchEnvPool(
          model, nbatch=2, nthread=0, numa_policy="pin", cpu_ids=[]
      )

  def test_cpu_ids_without_pin_rejected(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaises(ValueError):
      batch_env.BatchEnvPool(model, nbatch=2, nthread=2, cpu_ids=[0, 1])

  def test_negative_cpu_id_rejected(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaises(ValueError):
      batch_env.BatchEnvPool(
          model, nbatch=2, nthread=2, numa_policy="pin", cpu_ids=[0, -1]
      )

  def test_partitioned_without_partitions_rejected(self):
    # Phase 2 implements 'partitioned', but it requires a partitions list.
    # (Full partitioned behavior is covered by PartitionedNumaPolicyTest.)
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaisesRegex(ValueError, "requires a partitions list"):
      batch_env.BatchEnvPool(
          model, nbatch=2, nthread=2, numa_policy="partitioned"
      )

  def test_unknown_policy_rejected(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaises(ValueError):
      batch_env.BatchEnvPool(model, nbatch=2, nthread=2, numa_policy="numa")

  def test_worker_affinities_off_policy_is_empty(self):
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(model, nbatch=2, nthread=2) as pool:
      self.assertEqual(pool.worker_affinities, ())

  def test_worker_affinities_nonlinux_reports_fallback(self):
    if _sys.platform == "linux":
      self.skipTest("Linux has an affinity API; fallback path is not exercised")
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    # On macOS / Windows, numa_policy='pin' is accepted but the platform
    # has no affinity API. worker_affinities must be empty so callers can
    # detect the fallback rather than silently assuming pinning worked.
    with batch_env.BatchEnvPool(
        model, nbatch=2, nthread=2, numa_policy="pin", cpu_ids=[0, 1]
    ) as pool:
      self.assertEqual(pool.numa_policy, "pin")
      self.assertEqual(pool.worker_affinities, ())

  def test_linux_pin_actually_pins_workers(self):
    """On Linux, verify each worker's observed mask matches the request.

    This is the hard Phase 1 acceptance criterion ("worker fixed on
    requested CPU"). We read the OS-reported affinity via
    ``pool.worker_affinities`` and assert it is the singleton set
    ``{cpu_ids[i]}`` for every worker.
    """
    if _sys.platform != "linux":
      self.skipTest("affinity API is Linux-only")
    allowed = _linux_allowed_cpus()
    if len(allowed) < 2:
      self.skipTest(
          f"need at least 2 online CPUs, cgroup allows {allowed}"
      )
    cpu_ids = [allowed[0], allowed[1]]
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with batch_env.BatchEnvPool(
        model,
        nbatch=4,
        nthread=2,
        numa_policy="pin",
        cpu_ids=cpu_ids,
    ) as pool:
      affinities = pool.worker_affinities
      self.assertLen(affinities, 2)
      self.assertEqual(affinities[0], (cpu_ids[0],))
      self.assertEqual(affinities[1], (cpu_ids[1],))

  def test_linux_pin_invalid_cpu_raises(self):
    """On Linux, requesting a CPU the kernel rejects must fail loud."""
    if _sys.platform != "linux":
      self.skipTest("affinity API is Linux-only")
    # Pick a CPU id that is (almost certainly) not in the process's
    # allowed set. sched_setaffinity rejects requests whose mask has no
    # intersection with the cpuset. CPU_SETSIZE - 1 is 1023, well above
    # any realistic core count and outside any cgroup.
    bogus_cpu = 1023
    allowed = set(_linux_allowed_cpus())
    if bogus_cpu in allowed:
      self.skipTest(
          f"cpu {bogus_cpu} unexpectedly allowed; cannot exercise failure"
      )
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    with self.assertRaisesRegex(
        ValueError, "pthread_setaffinity_np|worker"
    ):
      batch_env.BatchEnvPool(
          model,
          nbatch=2,
          nthread=2,
          numa_policy="pin",
          cpu_ids=[bogus_cpu, bogus_cpu],
      )

  def test_pin_step_runs_and_matches_off_policy(self):
    """With policy='pin', step() must produce the same result as policy='off'.

    On non-Linux platforms this exercises the fallback path (affinity syscall
    is a no-op). On Linux it exercises the real pinning path when the
    requested CPUs are available.
    """
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    nbatch = 4
    nthread = 2
    nstep = 3
    ncontrol = model.nu

    rng = np.random.default_rng(0)
    state0 = _rand_state(model, nbatch, rng)
    control = rng.uniform(size=(nbatch, nstep, ncontrol))

    with batch_env.BatchEnvPool(model, nbatch=nbatch, nthread=nthread) as pool:
      s_ref, sd_ref = pool.step(
          state0, nstep=nstep, control=control, return_sensor=True
      )

    # On Linux, pick two CPUs actually allowed to the process (cgroup
    # or taskset restrictions can shrink this set). On other platforms
    # any small ids work because the syscall is a no-op.
    if _sys.platform == "linux":
      allowed = _linux_allowed_cpus()
      if len(allowed) < 2:
        self.skipTest(
            f"need at least 2 online CPUs, cgroup allows {allowed}"
        )
      cpu_ids = [allowed[0], allowed[1]]
    else:
      cpu_ids = [0, 1]

    with batch_env.BatchEnvPool(
        model,
        nbatch=nbatch,
        nthread=nthread,
        numa_policy="pin",
        cpu_ids=cpu_ids,
    ) as pool:
      self.assertEqual(pool.numa_policy, "pin")
      self.assertEqual(pool.cpu_ids, tuple(cpu_ids))
      s_pin, sd_pin = pool.step(
          state0, nstep=nstep, control=control, return_sensor=True
      )

    np.testing.assert_allclose(s_pin, s_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sd_pin, sd_ref, atol=1e-12, rtol=0)


class PartitionedNumaPolicyTest(absltest.TestCase):
  """Phase 2 numa_policy='partitioned': per-partition pools + first-touch."""

  def _model(self):
    return mujoco.MjModel.from_xml_string(TEST_XML)

  # ---- construction validation (pure ValueError, no threads needed) ----

  def test_partitioned_requires_partitions(self):
    with self.assertRaisesRegex(ValueError, "requires a partitions list"):
      batch_env.BatchEnvPool(self._model(), nbatch=4, numa_policy="partitioned")

  def test_partitions_must_start_at_zero(self):
    parts = [{"env_start": 1, "env_end": 4, "cpu_ids": [0], "nthread": 1}]
    with self.assertRaisesRegex(ValueError, "must start at env 0"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned", partitions=parts
      )

  def test_partitions_must_cover_nbatch(self):
    parts = [{"env_start": 0, "env_end": 3, "cpu_ids": [0], "nthread": 1}]
    with self.assertRaisesRegex(ValueError, "must cover"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned", partitions=parts
      )

  def test_partitions_must_be_contiguous(self):
    # gap between env 2 and env 3.
    parts = [
        {"env_start": 0, "env_end": 2, "cpu_ids": [0], "nthread": 1},
        {"env_start": 3, "env_end": 4, "cpu_ids": [1], "nthread": 1},
    ]
    with self.assertRaisesRegex(ValueError, "contiguous"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned", partitions=parts
      )

  def test_partitions_must_not_overlap(self):
    # env 1..2 covered by both partitions.
    parts = [
        {"env_start": 0, "env_end": 2, "cpu_ids": [0], "nthread": 1},
        {"env_start": 1, "env_end": 4, "cpu_ids": [1], "nthread": 1},
    ]
    with self.assertRaisesRegex(ValueError, "contiguous"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned", partitions=parts
      )

  def test_partition_requires_positive_nthread(self):
    parts = [{"env_start": 0, "env_end": 4, "cpu_ids": [], "nthread": 0}]
    with self.assertRaisesRegex(ValueError, "requires nthread > 0"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned", partitions=parts
      )

  def test_partition_cpu_ids_length_mismatch(self):
    parts = [{"env_start": 0, "env_end": 4, "cpu_ids": [0], "nthread": 2}]
    with self.assertRaisesRegex(ValueError, "cpu_ids length"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned", partitions=parts
      )

  def test_partition_negative_cpu_id(self):
    parts = [{"env_start": 0, "env_end": 4, "cpu_ids": [-1], "nthread": 1}]
    with self.assertRaisesRegex(ValueError, "non-negative"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned", partitions=parts
      )

  def test_cpu_ids_and_partitions_mutually_exclusive(self):
    parts = [{"env_start": 0, "env_end": 4, "cpu_ids": [0], "nthread": 1}]
    with self.assertRaisesRegex(ValueError, "top-level cpu_ids"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned",
          cpu_ids=[0], partitions=parts,
      )

  def test_partitions_rejected_for_off_policy(self):
    parts = [{"env_start": 0, "env_end": 4, "cpu_ids": [0], "nthread": 1}]
    with self.assertRaisesRegex(ValueError, "only valid with"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, nthread=2, partitions=parts
      )

  def test_partition_dict_bad_keys(self):
    parts = [{"env_start": 0, "env_end": 4, "cpu_ids": [0]}]  # missing nthread
    with self.assertRaisesRegex(ValueError, "must be a dict with keys"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned", partitions=parts
      )

  # ---- pinning (Linux only) ----

  def test_partitioned_worker_affinities(self):
    if _sys.platform != "linux":
      self.skipTest("affinity API is Linux-only")
    allowed = _linux_allowed_cpus()
    if len(allowed) < 4:
      self.skipTest(f"need >=4 online CPUs, cgroup allows {allowed}")
    c = allowed[:4]
    parts = [
        {"env_start": 0, "env_end": 4, "cpu_ids": [c[0], c[1]], "nthread": 2},
        {"env_start": 4, "env_end": 8, "cpu_ids": [c[2], c[3]], "nthread": 2},
    ]
    with batch_env.BatchEnvPool(
        self._model(), nbatch=8, numa_policy="partitioned", partitions=parts
    ) as pool:
      self.assertEqual(pool.numa_policy, "partitioned")
      self.assertEqual(pool.nthread, 4)
      # Affinities are concatenated across partitions in order.
      self.assertEqual(
          pool.worker_affinities,
          ((c[0],), (c[1],), (c[2],), (c[3],)),
      )
      # partitions property reflects the normalized plan.
      self.assertLen(pool.partitions, 2)
      self.assertEqual(pool.partitions[0]["env_start"], 0)
      self.assertEqual(pool.partitions[0]["env_end"], 4)
      self.assertEqual(pool.partitions[1]["cpu_ids"], (c[2], c[3]))

  def test_partitioned_two_numa_nodes(self):
    """Pin two partitions onto two different NUMA nodes when available."""
    if _sys.platform != "linux":
      self.skipTest("affinity API is Linux-only")
    allowed = set(_linux_allowed_cpus())
    # node0 low ids, node1 typically starts at cores-per-socket. Use a couple
    # of well-separated ids and skip if the process isn't allowed on them.
    want = [0, 1, 80, 81]
    if not set(want).issubset(allowed):
      self.skipTest(f"cpus {want} not all allowed; got {sorted(allowed)[:8]}...")
    parts = [
        {"env_start": 0, "env_end": 8, "cpu_ids": [0, 1], "nthread": 2},
        {"env_start": 8, "env_end": 16, "cpu_ids": [80, 81], "nthread": 2},
    ]
    with batch_env.BatchEnvPool(
        self._model(), nbatch=16, numa_policy="partitioned", partitions=parts
    ) as pool:
      self.assertEqual(
          pool.worker_affinities, ((0,), (1,), (80,), (81,))
      )

  def test_partitioned_invalid_cpu_raises(self):
    if _sys.platform != "linux":
      self.skipTest("affinity API is Linux-only")
    bogus = 1023
    if bogus in set(_linux_allowed_cpus()):
      self.skipTest(f"cpu {bogus} unexpectedly allowed")
    parts = [{"env_start": 0, "env_end": 4, "cpu_ids": [bogus], "nthread": 1}]
    with self.assertRaisesRegex(ValueError, "pthread_setaffinity_np|worker"):
      batch_env.BatchEnvPool(
          self._model(), nbatch=4, numa_policy="partitioned", partitions=parts
      )

  # ---- numerical equivalence (core correctness gate) ----

  def _partition_cpu_ids(self, nper):
    """Pick real CPU ids for `nper`-per-partition x 2 partitions, or None."""
    if _sys.platform != "linux":
      return [list(range(nper)), list(range(nper, 2 * nper))]
    allowed = _linux_allowed_cpus()
    if len(allowed) < 2 * nper:
      return None
    return [allowed[:nper], allowed[nper : 2 * nper]]

  def test_partitioned_step_matches_off(self):
    model = self._model()
    nbatch, nstep = 8, 3
    rng = np.random.default_rng(1)
    state0 = _rand_state(model, nbatch, rng)  # per-env distinct state
    control = rng.uniform(size=(nbatch, nstep, model.nu))

    with batch_env.BatchEnvPool(model, nbatch=nbatch, nthread=4) as pool:
      s_ref, sd_ref = pool.step(
          state0, nstep=nstep, control=control, return_sensor=True
      )

    cpus = self._partition_cpu_ids(2)
    if cpus is None:
      self.skipTest("need >=4 online CPUs")
    parts = [
        {"env_start": 0, "env_end": 4, "cpu_ids": cpus[0], "nthread": 2},
        {"env_start": 4, "env_end": 8, "cpu_ids": cpus[1], "nthread": 2},
    ]
    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, numa_policy="partitioned", partitions=parts
    ) as pool:
      s_p, sd_p = pool.step(
          state0, nstep=nstep, control=control, return_sensor=True
      )
    np.testing.assert_allclose(s_p, s_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sd_p, sd_ref, atol=1e-12, rtol=0)

  def test_partitioned_forward_matches_off(self):
    model = self._model()
    nbatch = 8
    rng = np.random.default_rng(2)
    state0 = _rand_state(model, nbatch, rng)

    with batch_env.BatchEnvPool(model, nbatch=nbatch, nthread=4) as pool:
      sd_ref = pool.forward(state0)

    cpus = self._partition_cpu_ids(2)
    if cpus is None:
      self.skipTest("need >=4 online CPUs")
    parts = [
        {"env_start": 0, "env_end": 5, "cpu_ids": cpus[0], "nthread": 2},
        {"env_start": 5, "env_end": 8, "cpu_ids": cpus[1], "nthread": 2},
    ]
    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, numa_policy="partitioned", partitions=parts
    ) as pool:
      sd_p = pool.forward(state0)
    np.testing.assert_allclose(sd_p, sd_ref, atol=1e-12, rtol=0)

  def test_partitioned_uneven_ranges(self):
    """Unequal partition sizes exercise per-partition chunk remainders."""
    model = self._model()
    nbatch, nstep = 10, 2
    rng = np.random.default_rng(3)
    state0 = _rand_state(model, nbatch, rng)
    control = rng.uniform(size=(nbatch, nstep, model.nu))

    with batch_env.BatchEnvPool(model, nbatch=nbatch, nthread=4) as pool:
      s_ref = pool.step(state0, nstep=nstep, control=control)

    cpus = self._partition_cpu_ids(2)
    if cpus is None:
      self.skipTest("need >=4 online CPUs")
    parts = [
        {"env_start": 0, "env_end": 3, "cpu_ids": cpus[0], "nthread": 2},
        {"env_start": 3, "env_end": 10, "cpu_ids": cpus[1], "nthread": 2},
    ]
    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, numa_policy="partitioned", partitions=parts
    ) as pool:
      s_p = pool.step(state0, nstep=nstep, control=control)
    np.testing.assert_allclose(s_p, s_ref, atol=1e-12, rtol=0)

  def test_single_partition_degenerate(self):
    """A single partition spanning [0, nbatch) equals off numerically."""
    model = self._model()
    nbatch, nstep = 6, 3
    rng = np.random.default_rng(4)
    state0 = _rand_state(model, nbatch, rng)
    control = rng.uniform(size=(nbatch, nstep, model.nu))

    with batch_env.BatchEnvPool(model, nbatch=nbatch, nthread=2) as pool:
      s_ref = pool.step(state0, nstep=nstep, control=control)

    if _sys.platform == "linux":
      allowed = _linux_allowed_cpus()
      if len(allowed) < 2:
        self.skipTest("need >=2 online CPUs")
      cpu = allowed[:2]
    else:
      cpu = [0, 1]
    parts = [{"env_start": 0, "env_end": nbatch, "cpu_ids": cpu, "nthread": 2}]
    with batch_env.BatchEnvPool(
        model, nbatch=nbatch, numa_policy="partitioned", partitions=parts
    ) as pool:
      self.assertEqual(pool.nthread, 2)
      s_p = pool.step(state0, nstep=nstep, control=control)
    np.testing.assert_allclose(s_p, s_ref, atol=1e-12, rtol=0)

  def test_first_touch_flag_roundtrips(self):
    model = self._model()
    cpus = self._partition_cpu_ids(1)
    if cpus is None:
      self.skipTest("need >=2 online CPUs")
    parts = [
        {"env_start": 0, "env_end": 2, "cpu_ids": cpus[0], "nthread": 1},
        {"env_start": 2, "env_end": 4, "cpu_ids": cpus[1], "nthread": 1},
    ]
    for ft in (True, False):
      with batch_env.BatchEnvPool(
          model, nbatch=4, numa_policy="partitioned",
          partitions=parts, first_touch=ft,
      ) as pool:
        self.assertEqual(pool.first_touch, ft)
        # step still runs correctly regardless of allocation site.
        rng = np.random.default_rng(5)
        state0 = _rand_state(model, 4, rng)
        pool.step(state0, nstep=1)


if __name__ == "__main__":
  absltest.main()
