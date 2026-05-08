// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Persistent executor for reset-lifecycle domain-randomized rollouts.
// Owns a per-environment mjModel pool and per-thread mjData workers, and
// provides:
//   * rollout(...)        — full-batch stepping over the owned model pool
//   * reset_subset(...)   — fused sparse reset with optional per-env model
//                           patching and selective mj_setConst refresh.
//
// The rollout kernel mirrors the one in rollout.cc. The subset-reset kernel
// mirrors the single-forward kernel in batch_forward_bind.cc, indexed by a
// caller-supplied env_ids array.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include "errors.h"
#include "raw.h"
#include "structs.h"
#include "threadpool.h"
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mujoco::python {

namespace {

namespace py = ::pybind11;
using PyCArray = py::array_t<mjtNum, py::array::c_style>;
using PyIArray = py::array_t<int, py::array::c_style>;

void KeepAlivePyObjectCapsuleDestructor(PyObject* pyobj) {
  py::gil_scoped_acquire gil;
  delete static_cast<py::object*>(PyCapsule_GetPointer(pyobj, nullptr));
}

// ===================================================================
// Field registry — supported reset-lifecycle randomization fields.
// ===================================================================

enum class FieldId {
  kBodyMass,
  kBodyIpos,
  kBodyIquat,
  kBodyInertia,
  kGeomFriction,
  kDofArmature,
  kGravity,
  kKp,
  kKd,
};

struct FieldSpec {
  const char* name;
  FieldId id;
  bool needs_refresh;  // needs mj_setConst after writing
};

constexpr FieldSpec kFieldSpecs[] = {
    {"body_mass",     FieldId::kBodyMass,     true},
    {"body_ipos",     FieldId::kBodyIpos,     true},
    {"body_iquat",    FieldId::kBodyIquat,    true},
    {"body_inertia",  FieldId::kBodyInertia,  true},
    {"dof_armature",  FieldId::kDofArmature,  true},
    {"gravity",       FieldId::kGravity,      false},
    {"geom_friction", FieldId::kGeomFriction, false},
    {"kp",            FieldId::kKp,           false},
    {"kd",            FieldId::kKd,           false},
};

// Element count of the field for a given model.
int FieldSize(FieldId id, const raw::MjModel* m) {
  switch (id) {
    case FieldId::kBodyMass:     return m->nbody;
    case FieldId::kBodyIpos:     return 3 * m->nbody;
    case FieldId::kBodyIquat:    return 4 * m->nbody;
    case FieldId::kBodyInertia:  return 3 * m->nbody;
    case FieldId::kDofArmature:  return m->nv;
    case FieldId::kGravity:      return 3;
    case FieldId::kGeomFriction: return 3 * m->ngeom;
    case FieldId::kKp:           return m->nu;
    case FieldId::kKd:           return m->nu;
  }
  return 0;
}

int FieldComponentWidth(FieldId id) {
  switch (id) {
    case FieldId::kBodyIpos:
    case FieldId::kBodyInertia:
    case FieldId::kGravity:
    case FieldId::kGeomFriction:
      return 3;
    case FieldId::kBodyIquat:
      return 4;
    case FieldId::kBodyMass:
    case FieldId::kDofArmature:
    case FieldId::kKp:
    case FieldId::kKd:
      return 1;
  }
  return 0;
}

int FieldIndexCount(FieldId id, const raw::MjModel* m) {
  switch (id) {
    case FieldId::kBodyMass:
    case FieldId::kBodyIpos:
    case FieldId::kBodyIquat:
    case FieldId::kBodyInertia:
      return m->nbody;
    case FieldId::kDofArmature:
      return m->nv;
    case FieldId::kGravity:
      return 1;
    case FieldId::kGeomFriction:
      return m->ngeom;
    case FieldId::kKp:
    case FieldId::kKd:
      return m->nu;
  }
  return 0;
}

// Pointer to contiguous field storage in m.
mjtNum* ContiguousFieldPtr(FieldId id, raw::MjModel* m) {
  switch (id) {
    case FieldId::kBodyMass:     return m->body_mass;
    case FieldId::kBodyIpos:     return m->body_ipos;
    case FieldId::kBodyIquat:    return m->body_iquat;
    case FieldId::kBodyInertia:  return m->body_inertia;
    case FieldId::kDofArmature:  return m->dof_armature;
    case FieldId::kGravity:      return m->opt.gravity;
    case FieldId::kGeomFriction: return m->geom_friction;
    case FieldId::kKp:
    case FieldId::kKd:
      return nullptr;
  }
  return nullptr;
}

const mjtNum* ContiguousFieldPtr(FieldId id, const raw::MjModel* m) {
  return ContiguousFieldPtr(id, const_cast<raw::MjModel*>(m));
}

void CopyFieldOut(FieldId id, const raw::MjModel* m, mjtNum* dst) {
  switch (id) {
    case FieldId::kBodyMass:
    case FieldId::kBodyIpos:
    case FieldId::kBodyIquat:
    case FieldId::kBodyInertia:
    case FieldId::kDofArmature:
    case FieldId::kGravity:
    case FieldId::kGeomFriction:
      mju_copy(dst, ContiguousFieldPtr(id, m), FieldSize(id, m));
      return;

    case FieldId::kKp:
      for (int i = 0; i < m->nu; ++i) {
        dst[i] = m->actuator_gainprm[mjNGAIN * i];
      }
      return;

    case FieldId::kKd:
      for (int i = 0; i < m->nu; ++i) {
        dst[i] = -m->actuator_biasprm[mjNBIAS * i + 2];
      }
      return;
  }
}

void WriteField(FieldId id, raw::MjModel* m, const mjtNum* src) {
  switch (id) {
    case FieldId::kBodyMass:
    case FieldId::kBodyIpos:
    case FieldId::kBodyIquat:
    case FieldId::kBodyInertia:
    case FieldId::kDofArmature:
    case FieldId::kGravity:
    case FieldId::kGeomFriction:
      mju_copy(ContiguousFieldPtr(id, m), src, FieldSize(id, m));
      return;

    case FieldId::kKp:
      for (int i = 0; i < m->nu; ++i) {
        m->actuator_gainprm[mjNGAIN * i] = src[i];
        m->actuator_biasprm[mjNBIAS * i + 1] = -src[i];
      }
      return;

    case FieldId::kKd:
      for (int i = 0; i < m->nu; ++i) {
        m->actuator_biasprm[mjNBIAS * i + 2] = -src[i];
      }
      return;
  }
}

void CopyIndexedFieldOut(FieldId id, const raw::MjModel* m, int index,
                         mjtNum* dst) {
  int width = FieldComponentWidth(id);
  switch (id) {
    case FieldId::kBodyMass:
    case FieldId::kBodyIpos:
    case FieldId::kBodyIquat:
    case FieldId::kBodyInertia:
    case FieldId::kDofArmature:
    case FieldId::kGravity:
    case FieldId::kGeomFriction:
      mju_copy(dst, ContiguousFieldPtr(id, m) + static_cast<size_t>(index) * width,
               width);
      return;

    case FieldId::kKp:
      dst[0] = m->actuator_gainprm[mjNGAIN * index];
      return;

    case FieldId::kKd:
      dst[0] = -m->actuator_biasprm[mjNBIAS * index + 2];
      return;
  }
}

void WriteIndexedField(FieldId id, raw::MjModel* m, int index,
                       const mjtNum* src) {
  int width = FieldComponentWidth(id);
  switch (id) {
    case FieldId::kBodyMass:
    case FieldId::kBodyIpos:
    case FieldId::kBodyIquat:
    case FieldId::kBodyInertia:
    case FieldId::kDofArmature:
    case FieldId::kGravity:
    case FieldId::kGeomFriction:
      mju_copy(ContiguousFieldPtr(id, m) + static_cast<size_t>(index) * width, src,
               width);
      return;

    case FieldId::kKp:
      m->actuator_gainprm[mjNGAIN * index] = src[0];
      m->actuator_biasprm[mjNBIAS * index + 1] = -src[0];
      return;

    case FieldId::kKd:
      m->actuator_biasprm[mjNBIAS * index + 2] = -src[0];
      return;
  }
}

const FieldSpec* LookupField(const std::string& name) {
  for (const FieldSpec& spec : kFieldSpecs) {
    if (name == spec.name) return &spec;
  }
  return nullptr;
}

const raw::MjModel* CastMjModelOrThrow(py::handle obj, const char* arg_name,
                                       std::optional<py::ssize_t> index =
                                           std::nullopt) {
  try {
    return obj.cast<const MjModelWrapper*>()->get();
  } catch (const py::cast_error&) {
    std::ostringstream msg;
    msg << arg_name;
    if (index.has_value()) {
      msg << "[" << *index << "]";
    }
    msg << " must be an MjModel";
    throw py::type_error(msg.str());
  }
}

bool SameModelDataLayout(const raw::MjModel* lhs, const raw::MjModel* rhs) {
  if (mj_stateSize(lhs, mjSTATE_FULLPHYSICS) !=
      mj_stateSize(rhs, mjSTATE_FULLPHYSICS)) {
    return false;
  }

#define MJ_MODEL_FIELD_EQ(name) \
  if (lhs->name != rhs->name) return false
  MJ_MODEL_FIELD_EQ(nq);
  MJ_MODEL_FIELD_EQ(nv);
  MJ_MODEL_FIELD_EQ(na);
  MJ_MODEL_FIELD_EQ(nu);
  MJ_MODEL_FIELD_EQ(nbody);
  MJ_MODEL_FIELD_EQ(nbvh);
  MJ_MODEL_FIELD_EQ(nbvhdynamic);
  MJ_MODEL_FIELD_EQ(njnt);
  MJ_MODEL_FIELD_EQ(nM);
  MJ_MODEL_FIELD_EQ(nC);
  MJ_MODEL_FIELD_EQ(nD);
  MJ_MODEL_FIELD_EQ(ngeom);
  MJ_MODEL_FIELD_EQ(nsite);
  MJ_MODEL_FIELD_EQ(ncam);
  MJ_MODEL_FIELD_EQ(nlight);
  MJ_MODEL_FIELD_EQ(nflexvert);
  MJ_MODEL_FIELD_EQ(nflexedge);
  MJ_MODEL_FIELD_EQ(nflexelem);
  MJ_MODEL_FIELD_EQ(nJfe);
  MJ_MODEL_FIELD_EQ(nJfv);
  MJ_MODEL_FIELD_EQ(neq);
  MJ_MODEL_FIELD_EQ(ntendon);
  MJ_MODEL_FIELD_EQ(nJten);
  MJ_MODEL_FIELD_EQ(nwrap);
  MJ_MODEL_FIELD_EQ(nsensordata);
  MJ_MODEL_FIELD_EQ(nmocap);
  MJ_MODEL_FIELD_EQ(nplugin);
  MJ_MODEL_FIELD_EQ(ntree);
  MJ_MODEL_FIELD_EQ(nhistory);
  MJ_MODEL_FIELD_EQ(nJmom);
  MJ_MODEL_FIELD_EQ(nuserdata);
  MJ_MODEL_FIELD_EQ(npluginstate);
  MJ_MODEL_FIELD_EQ(narena);
  MJ_MODEL_FIELD_EQ(nbuffer);
#undef MJ_MODEL_FIELD_EQ

  return true;
}

std::vector<const raw::MjModel*> NormalizeModelsOrThrow(py::object model_obj,
                                                        int nbatch) {
  if (py::isinstance<MjModelWrapper>(model_obj)) {
    const raw::MjModel* model = CastMjModelOrThrow(model_obj, "model");
    return std::vector<const raw::MjModel*>(nbatch, model);
  }

  py::sequence model_seq;
  try {
    model_seq = model_obj.cast<py::sequence>();
  } catch (const py::cast_error&) {
    throw py::type_error("model must be an MjModel or a sequence of MjModel");
  }

  py::ssize_t nmodel = py::len(model_seq);
  if (nmodel <= 0) {
    throw py::value_error("model sequence must not be empty");
  }
  if (nmodel != 1 && nmodel != nbatch) {
    std::ostringstream msg;
    msg << "model sequence must have length 1 or nbatch=" << nbatch
        << ", got " << nmodel;
    throw py::value_error(msg.str());
  }

  std::vector<const raw::MjModel*> models;
  models.reserve(nmodel == 1 ? nbatch : nmodel);
  for (py::ssize_t i = 0; i < nmodel; ++i) {
    models.push_back(CastMjModelOrThrow(model_seq[i], "model", i));
  }

  for (py::ssize_t i = 1; i < nmodel; ++i) {
    if (!SameModelDataLayout(models[0], models[i])) {
      std::ostringstream msg;
      msg << "models are not compatible: model[0] and model[" << i << "]";
      throw py::value_error(msg.str());
    }
  }

  if (nmodel == 1) {
    models.resize(nbatch, models[0]);
  }
  return models;
}

std::string SupportedFieldsString() {
  std::ostringstream msg;
  msg << "{";
  bool first = true;
  for (const FieldSpec& spec : kFieldSpecs) {
    if (!first) msg << ", ";
    msg << spec.name;
    first = false;
  }
  msg << "}";
  return msg.str();
}

const FieldSpec& LookupFieldOrThrow(const std::string& name) {
  const FieldSpec* spec = LookupField(name);
  if (!spec) {
    std::ostringstream msg;
    msg << "Unknown field '" << name
        << "'. Supported: " << SupportedFieldsString();
    throw py::value_error(msg.str());
  }
  return *spec;
}

void ValidateEnvIdOrThrow(int env_id, int nbatch) {
  if (env_id < 0 || env_id >= nbatch) {
    throw py::value_error("env_id out of range");
  }
}

void ValidateFieldIndexOrThrow(int index, int index_count,
                               const std::string& name) {
  if (index < 0 || index >= index_count) {
    std::ostringstream msg;
    msg << "Index " << index << " is out of range for field '" << name
        << "' with size " << index_count;
    throw py::value_error(msg.str());
  }
}

enum class HfieldAlignment {
  kYaw,
  kWorld,
  kBody,
};

enum class HfieldSampleOutput {
  kHeight,
  kClearance,
};

HfieldAlignment ParseHfieldAlignmentOrThrow(const std::string& alignment) {
  if (alignment == "yaw") return HfieldAlignment::kYaw;
  if (alignment == "world" || alignment == "none") {
    return HfieldAlignment::kWorld;
  }
  if (alignment == "body" || alignment == "full") {
    return HfieldAlignment::kBody;
  }
  std::ostringstream msg;
  msg << "alignment must be one of {'yaw', 'world', 'none', 'body', 'full'}, "
      << "got '" << alignment << "'";
  throw py::value_error(msg.str());
}

HfieldSampleOutput ParseHfieldOutputOrThrow(const std::string& output) {
  if (output == "height" || output == "terrain_height") {
    return HfieldSampleOutput::kHeight;
  }
  if (output == "clearance") {
    return HfieldSampleOutput::kClearance;
  }
  std::ostringstream msg;
  msg << "output must be one of {'height', 'terrain_height', 'clearance'}, "
      << "got '" << output << "'";
  throw py::value_error(msg.str());
}

void ValidateHfieldSamplerIdsOrThrow(
    const std::vector<raw::MjModel*>& models, int hfield_geom_id,
    int frame_body_id) {
  for (size_t i = 0; i < models.size(); ++i) {
    const raw::MjModel* m = models[i];
    if (hfield_geom_id < 0 || hfield_geom_id >= m->ngeom) {
      std::ostringstream msg;
      msg << "hfield_geom_id=" << hfield_geom_id
          << " is out of range for model[" << i << "] with ngeom="
          << m->ngeom;
      throw py::value_error(msg.str());
    }
    if (m->geom_type[hfield_geom_id] != mjGEOM_HFIELD) {
      std::ostringstream msg;
      msg << "geom " << hfield_geom_id << " in model[" << i
          << "] is not an hfield geom";
      throw py::value_error(msg.str());
    }
    int hfield_id = m->geom_dataid[hfield_geom_id];
    if (hfield_id < 0 || hfield_id >= m->nhfield) {
      std::ostringstream msg;
      msg << "geom " << hfield_geom_id << " in model[" << i
          << "] does not reference a valid hfield";
      throw py::value_error(msg.str());
    }
    if (m->hfield_nrow[hfield_id] < 2 || m->hfield_ncol[hfield_id] < 2) {
      std::ostringstream msg;
      msg << "hfield " << hfield_id << " in model[" << i
          << "] must be at least 2x2";
      throw py::value_error(msg.str());
    }
    if (frame_body_id < 0 || frame_body_id >= m->nbody) {
      std::ostringstream msg;
      msg << "frame_body_id=" << frame_body_id
          << " is out of range for model[" << i << "] with nbody="
          << m->nbody;
      throw py::value_error(msg.str());
    }
  }
}

mjtNum SampleHfieldLocalHeight(const raw::MjModel* m, int hfield_id,
                               mjtNum local_x, mjtNum local_y) {
  int nrow = m->hfield_nrow[hfield_id];
  int ncol = m->hfield_ncol[hfield_id];
  const mjtNum* size = m->hfield_size + 4 * hfield_id;
  const float* data = m->hfield_data + m->hfield_adr[hfield_id];

  mjtNum sx = (local_x + size[0]) / (2 * size[0]) * (ncol - 1);
  mjtNum sy = (local_y + size[1]) / (2 * size[1]) * (nrow - 1);
  sx = std::min<mjtNum>(std::max<mjtNum>(sx, 0), ncol - 1);
  sy = std::min<mjtNum>(std::max<mjtNum>(sy, 0), nrow - 1);

  int c0 = static_cast<int>(std::floor(sx));
  int r0 = static_cast<int>(std::floor(sy));
  int c1 = std::min(c0 + 1, ncol - 1);
  int r1 = std::min(r0 + 1, nrow - 1);
  mjtNum wx = sx - c0;
  mjtNum wy = sy - r0;

  mjtNum h00 = data[r0 * ncol + c0];
  mjtNum h10 = data[r0 * ncol + c1];
  mjtNum h01 = data[r1 * ncol + c0];
  mjtNum h11 = data[r1 * ncol + c1];
  mjtNum h0 = h00 * (1 - wx) + h10 * wx;
  mjtNum h1 = h01 * (1 - wx) + h11 * wx;
  return (h0 * (1 - wy) + h1 * wy) * size[2];
}

// ===================================================================
// Kernels — mirror rollout.cc / batch_forward_bind.cc, operating over
// the pool's owned models/data.
// ===================================================================

// Multi-step stepping over envs [start_roll, end_roll). Writes the FINAL
// state for each env to (nbatch, nstate) output. When sensordata_out is not
// null, also writes the final-step sensordata for each env to
// (nbatch, nsensordata). If post_step_forward_sensor is true, sensordata is
// recomputed by running one mj_forward on the final state before copying,
// matching the previous step()+forward() refresh path.
void _unsafe_step(const std::vector<const raw::MjModel*>& m, raw::MjData* d,
                  int start_roll, int end_roll, int nstep,
                  unsigned int control_spec, const mjtNum* state0,
                  const mjtNum* warmstart0, const mjtNum* control,
                  mjtNum* state_out, mjtNum* sensordata_out,
                  bool post_step_forward_sensor) {
  size_t nstate = static_cast<size_t>(mj_stateSize(m[0], mjSTATE_FULLPHYSICS));
  size_t ncontrol = static_cast<size_t>(mj_stateSize(m[0], control_spec));
  size_t nv = static_cast<size_t>(m[0]->nv);
  size_t nsensordata = static_cast<size_t>(m[0]->nsensordata);
  int nbody = m[0]->nbody, neq = m[0]->neq;

  if (!(control_spec & mjSTATE_CTRL)) {
    mju_zero(d->ctrl, m[0]->nu);
  }
  if (!(control_spec & mjSTATE_QFRC_APPLIED)) {
    mju_zero(d->qfrc_applied, nv);
  }
  if (!(control_spec & mjSTATE_XFRC_APPLIED)) {
    mju_zero(d->xfrc_applied, 6 * nbody);
  }

  for (size_t r = start_roll; r < static_cast<size_t>(end_roll); r++) {
    if (!(control_spec & mjSTATE_MOCAP_POS)) {
      for (int i = 0; i < nbody; i++) {
        int id = m[r]->body_mocapid[i];
        if (id >= 0) mju_copy3(d->mocap_pos + 3 * id, m[r]->body_pos + 3 * i);
      }
    }
    if (!(control_spec & mjSTATE_MOCAP_QUAT)) {
      for (int i = 0; i < nbody; i++) {
        int id = m[r]->body_mocapid[i];
        if (id >= 0) mju_copy4(d->mocap_quat + 4 * id, m[r]->body_quat + 4 * i);
      }
    }
    if (!(control_spec & mjSTATE_EQ_ACTIVE)) {
      for (int i = 0; i < neq; i++) {
        d->eq_active[i] = m[r]->eq_active0[i];
      }
    }

    mj_setState(m[r], d, state0 + r * nstate, mjSTATE_FULLPHYSICS);

    if (warmstart0) {
      mju_copy(d->qacc_warmstart, warmstart0 + r * nv, nv);
    } else {
      mju_zero(d->qacc_warmstart, nv);
    }

    for (int i = 0; i < mjNWARNING; i++) {
      d->warning[i].number = 0;
    }

    for (size_t t = 0; t < static_cast<size_t>(nstep); t++) {
      bool nwarning = false;
      for (int i = 0; i < mjNWARNING; i++) {
        if (d->warning[i].number) {
          nwarning = true;
          break;
        }
      }
      if (nwarning) break;

      if (control) {
        size_t step = r * static_cast<size_t>(nstep) + t;
        mj_setState(m[r], d, control + step * ncontrol, control_spec);
      }

      mj_step(m[r], d);
    }

    // Write only the final state for this env.
    mj_getState(m[r], d, state_out + r * nstate, mjSTATE_FULLPHYSICS);
    if (sensordata_out) {
      if (post_step_forward_sensor) {
        mju_zero(d->ctrl, m[r]->nu);
        mju_zero(d->qfrc_applied, nv);
        mju_zero(d->xfrc_applied, 6 * nbody);
        for (int i = 0; i < nbody; i++) {
          int id = m[r]->body_mocapid[i];
          if (id >= 0) {
            mju_copy3(d->mocap_pos + 3 * id, m[r]->body_pos + 3 * i);
            mju_copy4(d->mocap_quat + 4 * id, m[r]->body_quat + 4 * i);
          }
        }
        for (int i = 0; i < neq; i++) {
          d->eq_active[i] = m[r]->eq_active0[i];
        }
        mj_setState(m[r], d, state_out + r * nstate, mjSTATE_FULLPHYSICS);
        mju_zero(d->qacc_warmstart, nv);
        for (int i = 0; i < mjNWARNING; i++) {
          d->warning[i].number = 0;
        }
        mj_forwardSkip(m[r], d, mjSTAGE_NONE, 0);
      }
      mju_copy(sensordata_out + r * nsensordata, d->sensordata, nsensordata);
    }
  }
}

void _unsafe_step_threaded(const std::vector<const raw::MjModel*>& m,
                           std::vector<raw::MjData*>& d, int nbatch, int nstep,
                           unsigned int control_spec, const mjtNum* state0,
                           const mjtNum* warmstart0, const mjtNum* control,
                           mjtNum* state_out, mjtNum* sensordata_out,
                           bool post_step_forward_sensor, ThreadPool* pool,
                           int chunk_size) {
  int nfulljobs = nbatch / chunk_size;
  int chunk_remainder = nbatch % chunk_size;
  int njobs = (chunk_remainder > 0) ? nfulljobs + 1 : nfulljobs;

  pool->ResetCount();

  for (int j = 0; j < nfulljobs; j++) {
    auto task = [=, &m, &d](void) {
      int id = pool->WorkerId();
      _unsafe_step(m, d[id], j * chunk_size, (j + 1) * chunk_size, nstep,
                   control_spec, state0, warmstart0, control, state_out,
                   sensordata_out, post_step_forward_sensor);
    };
    pool->Schedule(task);
  }
  if (chunk_remainder > 0) {
    auto task = [=, &m, &d](void) {
      _unsafe_step(m, d[pool->WorkerId()], nfulljobs * chunk_size,
                   nfulljobs * chunk_size + chunk_remainder, nstep,
                   control_spec, state0, warmstart0, control, state_out,
                   sensordata_out, post_step_forward_sensor);
    };
    pool->Schedule(task);
  }

  pool->WaitCount(njobs);
}

// Subset reset kernel. Iterates over env_ids[start, end), applies per-env
// setup (clears controls/wrench, copies mocap, resets eq_active), then sets
// the caller's initial state, optionally runs mj_setConst for envs flagged
// for refresh, runs mj_forwardSkip, and writes state/sensor outputs into
// rows k (NOT rows env_ids[k]).
void _unsafe_subset_reset_range(
    const std::vector<raw::MjModel*>& models, raw::MjData* d, const int* env_ids,
    int start, int end, const mjtNum* initial_state,
    const mjtNum* initial_warmstart, const uint8_t* needs_refresh_per_k,
    mjtNum* state_out, mjtNum* sensordata_out, int skipsensor) {
  const raw::MjModel* m0 = models[0];
  size_t nstate = static_cast<size_t>(mj_stateSize(m0, mjSTATE_FULLPHYSICS));
  size_t nv = static_cast<size_t>(m0->nv);
  int nbody = m0->nbody;
  int neq = m0->neq;
  size_t nsensordata = static_cast<size_t>(m0->nsensordata);

  for (int k = start; k < end; ++k) {
    int e = env_ids[k];
    raw::MjModel* me = models[e];

    // Selective refresh of model-level derived constants (body_subtreemass,
    // dof_M0, ...). Uses d as scratch, so must run before we set up d.
    if (needs_refresh_per_k && needs_refresh_per_k[k]) {
      mj_setConst(me, d);
    }

    mju_zero(d->ctrl, me->nu);
    mju_zero(d->qfrc_applied, nv);
    mju_zero(d->xfrc_applied, 6 * nbody);

    for (int i = 0; i < nbody; i++) {
      int id = me->body_mocapid[i];
      if (id >= 0) {
        mju_copy3(d->mocap_pos + 3 * id, me->body_pos + 3 * i);
        mju_copy4(d->mocap_quat + 4 * id, me->body_quat + 4 * i);
      }
    }
    for (int i = 0; i < neq; i++) {
      d->eq_active[i] = me->eq_active0[i];
    }

    mj_setState(me, d, initial_state + static_cast<size_t>(k) * nstate,
                mjSTATE_FULLPHYSICS);

    if (initial_warmstart) {
      mju_copy(d->qacc_warmstart,
               initial_warmstart + static_cast<size_t>(k) * nv, nv);
    } else {
      mju_zero(d->qacc_warmstart, nv);
    }

    for (int i = 0; i < mjNWARNING; i++) {
      d->warning[i].number = 0;
    }

    mj_forwardSkip(me, d, mjSTAGE_NONE, skipsensor);

    if (state_out) {
      mj_getState(me, d, state_out + static_cast<size_t>(k) * nstate,
                  mjSTATE_FULLPHYSICS);
    }
    if (sensordata_out && !skipsensor) {
      mju_copy(sensordata_out + static_cast<size_t>(k) * nsensordata,
               d->sensordata, nsensordata);
    }
  }
}

void _unsafe_subset_reset_threaded(
    const std::vector<raw::MjModel*>& models, std::vector<raw::MjData*>& d,
    const int* env_ids, int n, const mjtNum* initial_state,
    const mjtNum* initial_warmstart, const uint8_t* needs_refresh_per_k,
    mjtNum* state_out, mjtNum* sensordata_out, int skipsensor, ThreadPool* pool,
    int chunk_size) {
  int nfulljobs = n / chunk_size;
  int chunk_remainder = n % chunk_size;
  int njobs = (chunk_remainder > 0) ? nfulljobs + 1 : nfulljobs;

  pool->ResetCount();

  for (int j = 0; j < nfulljobs; j++) {
    auto task = [=, &models, &d](void) {
      int id = pool->WorkerId();
      _unsafe_subset_reset_range(models, d[id], env_ids, j * chunk_size,
                                 (j + 1) * chunk_size, initial_state,
                                 initial_warmstart, needs_refresh_per_k,
                                 state_out, sensordata_out, skipsensor);
    };
    pool->Schedule(task);
  }
  if (chunk_remainder > 0) {
    auto task = [=, &models, &d](void) {
      _unsafe_subset_reset_range(
          models, d[pool->WorkerId()], env_ids, nfulljobs * chunk_size,
          nfulljobs * chunk_size + chunk_remainder, initial_state,
          initial_warmstart, needs_refresh_per_k, state_out, sensordata_out,
          skipsensor);
    };
    pool->Schedule(task);
  }

  pool->WaitCount(njobs);
}

// ===================================================================
// Helpers.
// ===================================================================

// Single mj_forward over envs [start, end) on the pool's owned models —
// no env_ids indirection, no model patching.
void _unsafe_full_forward_range(const std::vector<raw::MjModel*>& models,
                                raw::MjData* d, int start, int end,
                                const mjtNum* state0, const mjtNum* warmstart0,
                                mjtNum* sensordata_out, int skipsensor) {
  const raw::MjModel* m0 = models[0];
  size_t nstate = static_cast<size_t>(mj_stateSize(m0, mjSTATE_FULLPHYSICS));
  size_t nv = static_cast<size_t>(m0->nv);
  int nbody = m0->nbody;
  int neq = m0->neq;
  size_t nsensordata = static_cast<size_t>(m0->nsensordata);

  for (int r = start; r < end; ++r) {
    raw::MjModel* me = models[r];

    mju_zero(d->ctrl, me->nu);
    mju_zero(d->qfrc_applied, nv);
    mju_zero(d->xfrc_applied, 6 * nbody);

    for (int i = 0; i < nbody; i++) {
      int id = me->body_mocapid[i];
      if (id >= 0) {
        mju_copy3(d->mocap_pos + 3 * id, me->body_pos + 3 * i);
        mju_copy4(d->mocap_quat + 4 * id, me->body_quat + 4 * i);
      }
    }
    for (int i = 0; i < neq; i++) {
      d->eq_active[i] = me->eq_active0[i];
    }

    mj_setState(me, d, state0 + static_cast<size_t>(r) * nstate,
                mjSTATE_FULLPHYSICS);
    if (warmstart0) {
      mju_copy(d->qacc_warmstart,
               warmstart0 + static_cast<size_t>(r) * nv, nv);
    } else {
      mju_zero(d->qacc_warmstart, nv);
    }
    for (int i = 0; i < mjNWARNING; i++) {
      d->warning[i].number = 0;
    }

    mj_forwardSkip(me, d, mjSTAGE_NONE, skipsensor);

    if (!skipsensor) {
      mju_copy(sensordata_out + static_cast<size_t>(r) * nsensordata,
               d->sensordata, nsensordata);
    }
  }
}

void _unsafe_full_forward_threaded(const std::vector<raw::MjModel*>& models,
                                   std::vector<raw::MjData*>& d, int nbatch,
                                   const mjtNum* state0,
                                   const mjtNum* warmstart0,
                                   mjtNum* sensordata_out, int skipsensor,
                                   ThreadPool* pool, int chunk_size) {
  int nfulljobs = nbatch / chunk_size;
  int chunk_remainder = nbatch % chunk_size;
  int njobs = (chunk_remainder > 0) ? nfulljobs + 1 : nfulljobs;

  pool->ResetCount();

  for (int j = 0; j < nfulljobs; j++) {
    auto task = [=, &models, &d](void) {
      int id = pool->WorkerId();
      _unsafe_full_forward_range(models, d[id], j * chunk_size,
                                 (j + 1) * chunk_size, state0, warmstart0,
                                 sensordata_out, skipsensor);
    };
    pool->Schedule(task);
  }
  if (chunk_remainder > 0) {
    auto task = [=, &models, &d](void) {
      _unsafe_full_forward_range(
          models, d[pool->WorkerId()], nfulljobs * chunk_size,
          nfulljobs * chunk_size + chunk_remainder, state0, warmstart0,
          sensordata_out, skipsensor);
    };
    pool->Schedule(task);
  }

  pool->WaitCount(njobs);
}

// Site-Jacobian kernel. Per env: setState, run kinematics + comPos prefix
// (the minimum that mj_jacSite reads from), then emit jacp/jacr for every
// requested site into the contiguous (nbatch, nsite_query, 3, nv) outputs.
// Output pointers may be null when the corresponding flavor is not requested.
void _unsafe_jacobians_range(const std::vector<raw::MjModel*>& models,
                             raw::MjData* d, int start, int end,
                             const mjtNum* state0, const mjtNum* warmstart0,
                             const int* site_ids, int nsite_query,
                             mjtNum* jacp_out, mjtNum* jacr_out) {
  const raw::MjModel* m0 = models[0];
  size_t nstate = static_cast<size_t>(mj_stateSize(m0, mjSTATE_FULLPHYSICS));
  size_t nv = static_cast<size_t>(m0->nv);
  int nbody = m0->nbody;
  int neq = m0->neq;
  size_t per_env_block =
      static_cast<size_t>(nsite_query) * 3 * nv;

  for (int r = start; r < end; ++r) {
    raw::MjModel* me = models[r];

    mju_zero(d->ctrl, me->nu);
    mju_zero(d->qfrc_applied, nv);
    mju_zero(d->xfrc_applied, 6 * nbody);

    for (int i = 0; i < nbody; i++) {
      int id = me->body_mocapid[i];
      if (id >= 0) {
        mju_copy3(d->mocap_pos + 3 * id, me->body_pos + 3 * i);
        mju_copy4(d->mocap_quat + 4 * id, me->body_quat + 4 * i);
      }
    }
    for (int i = 0; i < neq; i++) {
      d->eq_active[i] = me->eq_active0[i];
    }

    mj_setState(me, d, state0 + static_cast<size_t>(r) * nstate,
                mjSTATE_FULLPHYSICS);
    if (warmstart0) {
      mju_copy(d->qacc_warmstart,
               warmstart0 + static_cast<size_t>(r) * nv, nv);
    } else {
      mju_zero(d->qacc_warmstart, nv);
    }
    for (int i = 0; i < mjNWARNING; i++) {
      d->warning[i].number = 0;
    }

    // Position-stage prefix that mj_jacSite reads:
    //   site_xpos        ← mj_kinematics
    //   subtree_com,cdof ← mj_comPos
    mj_kinematics(me, d);
    mj_comPos(me, d);

    size_t env_off = static_cast<size_t>(r) * per_env_block;
    for (int s = 0; s < nsite_query; ++s) {
      size_t out_off = env_off + static_cast<size_t>(s) * 3 * nv;
      mjtNum* jp = jacp_out ? jacp_out + out_off : nullptr;
      mjtNum* jr = jacr_out ? jacr_out + out_off : nullptr;
      mj_jacSite(me, d, jp, jr, site_ids[s]);
    }
  }
}

void _unsafe_jacobians_threaded(const std::vector<raw::MjModel*>& models,
                                std::vector<raw::MjData*>& d, int nbatch,
                                const mjtNum* state0, const mjtNum* warmstart0,
                                const int* site_ids, int nsite_query,
                                mjtNum* jacp_out, mjtNum* jacr_out,
                                ThreadPool* pool, int chunk_size) {
  int nfulljobs = nbatch / chunk_size;
  int chunk_remainder = nbatch % chunk_size;
  int njobs = (chunk_remainder > 0) ? nfulljobs + 1 : nfulljobs;

  pool->ResetCount();

  for (int j = 0; j < nfulljobs; j++) {
    auto task = [=, &models, &d](void) {
      int id = pool->WorkerId();
      _unsafe_jacobians_range(models, d[id], j * chunk_size,
                              (j + 1) * chunk_size, state0, warmstart0,
                              site_ids, nsite_query, jacp_out, jacr_out);
    };
    pool->Schedule(task);
  }
  if (chunk_remainder > 0) {
    auto task = [=, &models, &d](void) {
      _unsafe_jacobians_range(
          models, d[pool->WorkerId()], nfulljobs * chunk_size,
          nfulljobs * chunk_size + chunk_remainder, state0, warmstart0,
          site_ids, nsite_query, jacp_out, jacr_out);
    };
    pool->Schedule(task);
  }

  pool->WaitCount(njobs);
}

// Hfield bilinear sampling kernel. Per env: set full-physics state, run the
// kinematic prefix needed for body and geom world poses, transform the local
// offset pattern through the requested frame alignment, then sample the target
// hfield in geom-local coordinates. Boundary samples are clamped to the hfield
// extent.
void _unsafe_hfield_sample_range(
    const std::vector<raw::MjModel*>& models, raw::MjData* d, int start,
    int end, const mjtNum* state0, int hfield_geom_id,
    const mjtNum* offsets, int npoint, int frame_body_id,
    HfieldAlignment alignment, HfieldSampleOutput output, mjtNum* sample_out) {
  const raw::MjModel* m0 = models[0];
  size_t nstate = static_cast<size_t>(mj_stateSize(m0, mjSTATE_FULLPHYSICS));
  size_t nv = static_cast<size_t>(m0->nv);
  int nbody = m0->nbody;
  int neq = m0->neq;

  for (int r = start; r < end; ++r) {
    raw::MjModel* me = models[r];

    mju_zero(d->ctrl, me->nu);
    mju_zero(d->qfrc_applied, nv);
    mju_zero(d->xfrc_applied, 6 * nbody);

    for (int i = 0; i < nbody; i++) {
      int id = me->body_mocapid[i];
      if (id >= 0) {
        mju_copy3(d->mocap_pos + 3 * id, me->body_pos + 3 * i);
        mju_copy4(d->mocap_quat + 4 * id, me->body_quat + 4 * i);
      }
    }
    for (int i = 0; i < neq; i++) {
      d->eq_active[i] = me->eq_active0[i];
    }

    mj_setState(me, d, state0 + static_cast<size_t>(r) * nstate,
                mjSTATE_FULLPHYSICS);
    mju_zero(d->qacc_warmstart, nv);
    for (int i = 0; i < mjNWARNING; i++) {
      d->warning[i].number = 0;
    }

    mj_kinematics(me, d);

    int hfield_id = me->geom_dataid[hfield_geom_id];
    const mjtNum* frame_pos = d->xpos + 3 * frame_body_id;
    const mjtNum* frame_mat = d->xmat + 9 * frame_body_id;
    const mjtNum* geom_pos = d->geom_xpos + 3 * hfield_geom_id;
    const mjtNum* geom_mat = d->geom_xmat + 9 * hfield_geom_id;

    mjtNum cos_yaw = 1;
    mjtNum sin_yaw = 0;
    if (alignment == HfieldAlignment::kYaw) {
      mjtNum yaw = std::atan2(frame_mat[3], frame_mat[0]);
      cos_yaw = std::cos(yaw);
      sin_yaw = std::sin(yaw);
    }

    mjtNum* out = sample_out + static_cast<size_t>(r) * npoint;
    for (int p = 0; p < npoint; ++p) {
      mjtNum ox = offsets[2 * p];
      mjtNum oy = offsets[2 * p + 1];
      mjtNum world_x;
      mjtNum world_y;
      switch (alignment) {
        case HfieldAlignment::kYaw:
          world_x = frame_pos[0] + cos_yaw * ox - sin_yaw * oy;
          world_y = frame_pos[1] + sin_yaw * ox + cos_yaw * oy;
          break;
        case HfieldAlignment::kBody:
          world_x = frame_pos[0] + frame_mat[0] * ox + frame_mat[1] * oy;
          world_y = frame_pos[1] + frame_mat[3] * ox + frame_mat[4] * oy;
          break;
        case HfieldAlignment::kWorld:
          world_x = frame_pos[0] + ox;
          world_y = frame_pos[1] + oy;
          break;
      }

      mjtNum rel_x = world_x - geom_pos[0];
      mjtNum rel_y = world_y - geom_pos[1];
      mjtNum local_x = geom_mat[0] * rel_x + geom_mat[3] * rel_y;
      mjtNum local_y = geom_mat[1] * rel_x + geom_mat[4] * rel_y;
      mjtNum local_height =
          SampleHfieldLocalHeight(me, hfield_id, local_x, local_y);
      mjtNum terrain_z = geom_pos[2] + geom_mat[6] * local_x +
                         geom_mat[7] * local_y +
                         geom_mat[8] * local_height;
      out[p] = output == HfieldSampleOutput::kClearance
                   ? frame_pos[2] - terrain_z
                   : terrain_z;
    }
  }
}

void _unsafe_hfield_sample_threaded(
    const std::vector<raw::MjModel*>& models, std::vector<raw::MjData*>& d,
    int nbatch, const mjtNum* state0, int hfield_geom_id,
    const mjtNum* offsets, int npoint, int frame_body_id,
    HfieldAlignment alignment, HfieldSampleOutput output, mjtNum* sample_out,
    ThreadPool* pool, int chunk_size) {
  int nfulljobs = nbatch / chunk_size;
  int chunk_remainder = nbatch % chunk_size;
  int njobs = (chunk_remainder > 0) ? nfulljobs + 1 : nfulljobs;

  pool->ResetCount();

  for (int j = 0; j < nfulljobs; j++) {
    auto task = [=, &models, &d](void) {
      int id = pool->WorkerId();
      _unsafe_hfield_sample_range(
          models, d[id], j * chunk_size, (j + 1) * chunk_size, state0,
          hfield_geom_id, offsets, npoint, frame_body_id, alignment, output,
          sample_out);
    };
    pool->Schedule(task);
  }
  if (chunk_remainder > 0) {
    auto task = [=, &models, &d](void) {
      _unsafe_hfield_sample_range(
          models, d[pool->WorkerId()], nfulljobs * chunk_size,
          nfulljobs * chunk_size + chunk_remainder, state0, hfield_geom_id,
          offsets, npoint, frame_body_id, alignment, output, sample_out);
    };
    pool->Schedule(task);
  }

  pool->WaitCount(njobs);
}

mjtNum* optional_array_ptr(const std::optional<PyCArray>& arg, const char* name,
                           size_t expected) {
  if (!arg.has_value()) return nullptr;
  py::buffer_info info = arg->request();
  if (static_cast<size_t>(info.size) != expected) {
    std::ostringstream msg;
    msg << name << ".size should be " << expected << ", got " << info.size;
    throw py::value_error(msg.str());
  }
  return static_cast<mjtNum*>(info.ptr);
}

mjtNum* required_array_ptr(const PyCArray& arr, const char* name,
                           size_t expected) {
  py::buffer_info info = arr.request();
  if (static_cast<size_t>(info.size) != expected) {
    std::ostringstream msg;
    msg << name << ".size should be " << expected << ", got " << info.size;
    throw py::value_error(msg.str());
  }
  return static_cast<mjtNum*>(info.ptr);
}

// ===================================================================
// Pool.
// ===================================================================

class BatchEnvPool {
 public:
  BatchEnvPool(py::object model_obj, int nbatch, int nthread)
      : nbatch_(nbatch), nthread_(nthread) {
    if (nbatch_ <= 0) {
      throw py::value_error("nbatch must be positive");
    }
    if (nthread_ < 0) {
      throw py::value_error("nthread must be >= 0");
    }

    std::vector<const raw::MjModel*> src_models =
        NormalizeModelsOrThrow(model_obj, nbatch_);

    models_.resize(nbatch_, nullptr);
    for (int i = 0; i < nbatch_; ++i) {
      models_[i] = InterceptMjErrors(mj_copyModel)(nullptr, src_models[i]);
      if (models_[i] == nullptr) {
        // cleanup previously allocated models before throwing
        for (int j = 0; j < i; ++j) {
          mj_deleteModel(models_[j]);
          models_[j] = nullptr;
        }
        throw py::value_error("mj_copyModel failed while constructing pool");
      }
    }

    int ndata = nthread_ > 0 ? nthread_ : 1;
    worker_data_.resize(ndata, nullptr);
    for (int t = 0; t < ndata; ++t) {
      worker_data_[t] = mj_makeData(models_[0]);
      if (worker_data_[t] == nullptr) {
        for (int j = 0; j < t; ++j) {
          mj_deleteData(worker_data_[j]);
          worker_data_[j] = nullptr;
        }
        for (int j = 0; j < nbatch_; ++j) {
          mj_deleteModel(models_[j]);
          models_[j] = nullptr;
        }
        throw py::value_error("mj_makeData failed while constructing pool");
      }
    }

    if (nthread_ > 0) {
      pool_ = std::make_shared<ThreadPool>(nthread_);
    }

    // Cache sizes.
    nstate_ = mj_stateSize(models_[0], mjSTATE_FULLPHYSICS);
    nv_ = models_[0]->nv;
    nsensordata_ = models_[0]->nsensordata;
  }

  ~BatchEnvPool() {
    // shared_ptr release joins the thread pool before we tear down data.
    pool_.reset();
    for (raw::MjData* d : worker_data_) {
      if (d) mj_deleteData(d);
    }
    worker_data_.clear();
    for (raw::MjModel* m : models_) {
      if (m) mj_deleteModel(m);
    }
    models_.clear();
  }

  // -----------------------------------------------------------------
  // Multi-step stepping. Returns the FINAL state for each env, and optionally
  // the final-step sensordata. By default sensors are not copied, matching the
  // historical state-only return.
  //   state output: (nbatch, nstate)
  //   sensordata output: (nbatch, nsensordata), optional
  // -----------------------------------------------------------------
  py::object step(int nstep, unsigned int control_spec,
                  const PyCArray state0,
                  std::optional<const PyCArray> warmstart0,
                  std::optional<const PyCArray> control,
                  std::optional<int> chunk_size, bool return_sensor,
                  bool post_step_forward_sensor) {
    if (nstep < 1) {
      throw py::value_error("nstep must be >= 1");
    }
    if (state0.shape(0) != nbatch_) {
      std::ostringstream msg;
      msg << "state0.shape[0]=" << state0.shape(0)
          << " does not match pool nbatch=" << nbatch_;
      throw py::value_error(msg.str());
    }

    int ncontrol = mj_stateSize(models_[0], control_spec);

    size_t sz_state0 = static_cast<size_t>(nbatch_) * nstate_;
    size_t sz_warmstart = static_cast<size_t>(nbatch_) * nv_;
    size_t sz_control =
        static_cast<size_t>(nbatch_) * nstep * ncontrol;

    std::optional<PyCArray> state0_opt = state0;
    mjtNum* state0_ptr = optional_array_ptr(state0_opt, "state0", sz_state0);
    mjtNum* warmstart0_ptr =
        optional_array_ptr(warmstart0, "warmstart0", sz_warmstart);
    mjtNum* control_ptr = optional_array_ptr(control, "control", sz_control);

    PyCArray state_out({nbatch_, nstate_});
    mjtNum* state_ptr = static_cast<mjtNum*>(state_out.request().ptr);
    std::optional<PyCArray> sensordata_out;
    mjtNum* sensordata_ptr = nullptr;
    if (return_sensor) {
      sensordata_out.emplace(std::vector<py::ssize_t>{nbatch_, nsensordata_});
      sensordata_ptr =
          static_cast<mjtNum*>(sensordata_out->request().ptr);
    }

    std::vector<const raw::MjModel*> cmodels(nbatch_);
    for (int i = 0; i < nbatch_; ++i) cmodels[i] = models_[i];

    {
      py::gil_scoped_release no_gil;
      if (nthread_ > 0 && nbatch_ > 1) {
        int chunk = chunk_size.has_value()
                        ? *chunk_size
                        : std::max(1, nbatch_ / (10 * nthread_));
        InterceptMjErrors(_unsafe_step_threaded)(
            cmodels, worker_data_, nbatch_, nstep, control_spec, state0_ptr,
            warmstart0_ptr, control_ptr, state_ptr, sensordata_ptr,
            post_step_forward_sensor, pool_.get(), chunk);
      } else {
        InterceptMjErrors(_unsafe_step)(
            cmodels, worker_data_[0], 0, nbatch_, nstep, control_spec,
            state0_ptr, warmstart0_ptr, control_ptr, state_ptr,
            sensordata_ptr, post_step_forward_sensor);
      }
    }

    if (return_sensor) {
      return py::make_tuple(state_out, *sensordata_out);
    }
    return state_out;
  }

  // -----------------------------------------------------------------
  // Single mj_forward over all envs (no patching, no env_ids).
  //   sensordata output: (nbatch, nsensordata)
  // Equivalent to the old mujoco.batch_forward path, just owned by the
  // pool's models/data instead of caller-supplied lists.
  // -----------------------------------------------------------------
  PyCArray forward(const PyCArray state0,
                   std::optional<const PyCArray> warmstart0, bool skipsensor,
                   std::optional<int> chunk_size) {
    if (state0.shape(0) != nbatch_) {
      std::ostringstream msg;
      msg << "state0.shape[0]=" << state0.shape(0)
          << " does not match pool nbatch=" << nbatch_;
      throw py::value_error(msg.str());
    }

    size_t sz_state0 = static_cast<size_t>(nbatch_) * nstate_;
    size_t sz_warmstart = static_cast<size_t>(nbatch_) * nv_;
    std::optional<PyCArray> state0_opt = state0;
    mjtNum* state0_ptr = optional_array_ptr(state0_opt, "state0", sz_state0);
    mjtNum* warmstart0_ptr =
        optional_array_ptr(warmstart0, "warmstart0", sz_warmstart);

    PyCArray sensordata_out({nbatch_, nsensordata_});
    mjtNum* sensordata_ptr =
        static_cast<mjtNum*>(sensordata_out.request().ptr);

    {
      py::gil_scoped_release no_gil;
      if (nthread_ > 0 && nbatch_ > 1) {
        int chunk = chunk_size.has_value()
                        ? *chunk_size
                        : std::max(1, nbatch_ / (10 * nthread_));
        InterceptMjErrors(_unsafe_full_forward_threaded)(
            models_, worker_data_, nbatch_, state0_ptr, warmstart0_ptr,
            sensordata_ptr, skipsensor ? 1 : 0, pool_.get(), chunk);
      } else {
        InterceptMjErrors(_unsafe_full_forward_range)(
            models_, worker_data_[0], 0, nbatch_, state0_ptr, warmstart0_ptr,
            sensordata_ptr, skipsensor ? 1 : 0);
      }
    }

    return sensordata_out;
  }

  // -----------------------------------------------------------------
  // Site-Jacobian batch.
  //
  //   state0:    (nbatch, nstate)
  //   site_ids:  (K,) int32, in [0, nsite); same site set used for every env
  //   jacp/jacr: at least one must be true
  //
  // Returns (jacp_or_none, jacr_or_none); each is (nbatch, K, 3, nv).
  //
  // Internally re-runs mj_kinematics + mj_comPos per env on the worker's
  // mjData, then calls mj_jacSite for each site. No constraint or actuator
  // stages are evaluated.
  // -----------------------------------------------------------------
  py::tuple compute_site_jacobians(
      const PyCArray state0, const PyIArray site_ids, bool jacp, bool jacr,
      std::optional<const PyCArray> warmstart0,
      std::optional<int> chunk_size) {
    if (!jacp && !jacr) {
      throw py::value_error(
          "compute_site_jacobians: at least one of jacp / jacr must be True");
    }
    if (state0.ndim() != 2 || state0.shape(0) != nbatch_) {
      std::ostringstream msg;
      msg << "state0 must have shape (nbatch=" << nbatch_
          << ", nstate=" << nstate_ << "), got (";
      for (int i = 0; i < state0.ndim(); ++i) {
        msg << state0.shape(i) << (i + 1 < state0.ndim() ? ", " : "");
      }
      msg << ")";
      throw py::value_error(msg.str());
    }

    if (site_ids.ndim() != 1) {
      throw py::value_error("site_ids must be a 1-D array");
    }
    int nsite_query = static_cast<int>(site_ids.shape(0));
    if (nsite_query <= 0) {
      throw py::value_error("site_ids must be non-empty");
    }
    int nsite_model = models_[0]->nsite;
    const int* site_ids_ptr = site_ids.data();
    for (int s = 0; s < nsite_query; ++s) {
      int id = site_ids_ptr[s];
      if (id < 0 || id >= nsite_model) {
        std::ostringstream msg;
        msg << "site_ids[" << s << "]=" << id << " is out of range [0, "
            << nsite_model << ")";
        throw py::value_error(msg.str());
      }
    }

    size_t sz_state0 = static_cast<size_t>(nbatch_) * nstate_;
    size_t sz_warmstart = static_cast<size_t>(nbatch_) * nv_;
    std::optional<PyCArray> state0_opt = state0;
    mjtNum* state0_ptr = optional_array_ptr(state0_opt, "state0", sz_state0);
    mjtNum* warmstart0_ptr =
        optional_array_ptr(warmstart0, "warmstart0", sz_warmstart);

    std::vector<py::ssize_t> shape = {nbatch_, nsite_query, 3, nv_};
    py::object none = py::none();
    py::object jacp_obj = none;
    py::object jacr_obj = none;
    mjtNum* jacp_ptr = nullptr;
    mjtNum* jacr_ptr = nullptr;
    if (jacp) {
      PyCArray arr(shape);
      jacp_ptr = static_cast<mjtNum*>(arr.request().ptr);
      jacp_obj = std::move(arr);
    }
    if (jacr) {
      PyCArray arr(shape);
      jacr_ptr = static_cast<mjtNum*>(arr.request().ptr);
      jacr_obj = std::move(arr);
    }

    {
      py::gil_scoped_release no_gil;
      if (nthread_ > 0 && nbatch_ > 1) {
        int chunk = chunk_size.has_value()
                        ? *chunk_size
                        : std::max(1, nbatch_ / (10 * nthread_));
        InterceptMjErrors(_unsafe_jacobians_threaded)(
            models_, worker_data_, nbatch_, state0_ptr, warmstart0_ptr,
            site_ids_ptr, nsite_query, jacp_ptr, jacr_ptr, pool_.get(), chunk);
      } else {
        InterceptMjErrors(_unsafe_jacobians_range)(
            models_, worker_data_[0], 0, nbatch_, state0_ptr, warmstart0_ptr,
            site_ids_ptr, nsite_query, jacp_ptr, jacr_ptr);
      }
    }

    return py::make_tuple(jacp_obj, jacr_obj);
  }

  // -----------------------------------------------------------------
  // MuJoCo hfield bilinear sampler.
  //
  //   state0:          (nbatch, nstate)
  //   hfield_geom_id:  target geom; must be type=hfield in every model
  //   offsets:         (K, 2) local XY pattern attached to frame_body_id
  //   frame_body_id:   body providing attachment pose and clearance z
  //   alignment:
  //     "yaw"          rotate offsets by frame yaw only
  //     "world"/"none" keep offsets in world XY axes
  //     "body"/"full"  rotate offsets by the frame body's full XY columns
  //   output:
  //     "height"       sampled terrain world z
  //     "clearance"    frame body world z minus sampled terrain world z
  //
  // Returns (nbatch, K). Metadata stays in the pool-owned mjModel; the hot
  // path only sets state, runs kinematics, and samples hfield_data.
  // -----------------------------------------------------------------
  PyCArray sample_hfield_height(
      const PyCArray state0, int hfield_geom_id, const PyCArray offsets,
      int frame_body_id, const std::string& alignment,
      const std::string& output, std::optional<int> chunk_size) {
    if (state0.ndim() != 2 || state0.shape(0) != nbatch_ ||
        state0.shape(1) != nstate_) {
      std::ostringstream msg;
      msg << "state0 must have shape (nbatch=" << nbatch_
          << ", nstate=" << nstate_ << "), got (";
      for (int i = 0; i < state0.ndim(); ++i) {
        msg << state0.shape(i) << (i + 1 < state0.ndim() ? ", " : "");
      }
      msg << ")";
      throw py::value_error(msg.str());
    }
    if (offsets.ndim() != 2 || offsets.shape(1) != 2) {
      std::ostringstream msg;
      msg << "offsets must have shape (npoint, 2), got (";
      for (int i = 0; i < offsets.ndim(); ++i) {
        msg << offsets.shape(i) << (i + 1 < offsets.ndim() ? ", " : "");
      }
      msg << ")";
      throw py::value_error(msg.str());
    }
    int npoint = static_cast<int>(offsets.shape(0));
    if (npoint <= 0) {
      throw py::value_error("offsets must be non-empty");
    }
    if (chunk_size.has_value() && *chunk_size <= 0) {
      throw py::value_error("chunk_size must be positive");
    }

    HfieldAlignment alignment_enum =
        ParseHfieldAlignmentOrThrow(alignment);
    HfieldSampleOutput output_enum = ParseHfieldOutputOrThrow(output);
    ValidateHfieldSamplerIdsOrThrow(
        models_, hfield_geom_id, frame_body_id);

    size_t sz_state0 = static_cast<size_t>(nbatch_) * nstate_;
    size_t sz_offsets = static_cast<size_t>(npoint) * 2;
    std::optional<PyCArray> state0_opt = state0;
    mjtNum* state0_ptr = optional_array_ptr(state0_opt, "state0", sz_state0);
    mjtNum* offsets_ptr =
        required_array_ptr(offsets, "offsets", sz_offsets);

    PyCArray sample_out({nbatch_, npoint});
    mjtNum* sample_ptr = static_cast<mjtNum*>(sample_out.request().ptr);

    {
      py::gil_scoped_release no_gil;
      if (nthread_ > 0 && nbatch_ > 1) {
        int chunk = chunk_size.has_value()
                        ? *chunk_size
                        : std::max(1, nbatch_ / (10 * nthread_));
        InterceptMjErrors(_unsafe_hfield_sample_threaded)(
            models_, worker_data_, nbatch_, state0_ptr, hfield_geom_id,
            offsets_ptr, npoint, frame_body_id, alignment_enum, output_enum,
            sample_ptr, pool_.get(), chunk);
      } else {
        InterceptMjErrors(_unsafe_hfield_sample_range)(
            models_, worker_data_[0], 0, nbatch_, state0_ptr, hfield_geom_id,
            offsets_ptr, npoint, frame_body_id, alignment_enum, output_enum,
            sample_ptr);
      }
    }

    return sample_out;
  }

  // -----------------------------------------------------------------
  // Sparse reset with optional per-env model patching.
  //
  // randomization: dict[str, ndarray] where each ndarray has leading dim
  //                equal to len(env_ids) and trailing size equal to the
  //                field's element count for the base model.
  // -----------------------------------------------------------------
  py::tuple reset(py::array env_ids_arr, const PyCArray initial_state,
                  std::optional<py::dict> randomization,
                  std::optional<const PyCArray> initial_warmstart,
                  bool skipsensor, std::optional<int> chunk_size) {
    // Canonicalize env_ids to contiguous int32.
    py::module_ np = py::module_::import("numpy");
    py::object ids_obj = np.attr("ascontiguousarray")(
        env_ids_arr, py::arg("dtype") = np.attr("int32"));
    py::array_t<int, py::array::c_style> env_ids =
        ids_obj.cast<py::array_t<int, py::array::c_style>>();
    if (env_ids.ndim() != 1) {
      throw py::value_error("env_ids must be a 1-D array");
    }
    int n = static_cast<int>(env_ids.shape(0));

    PyCArray state_out({n, nstate_});
    PyCArray sensordata_out({n, nsensordata_});

    if (n == 0) {
      return py::make_tuple(state_out, sensordata_out);
    }

    const int* env_ids_ptr = env_ids.data();
    for (int k = 0; k < n; ++k) {
      int e = env_ids_ptr[k];
      if (e < 0 || e >= nbatch_) {
        std::ostringstream msg;
        msg << "env_ids[" << k << "]=" << e << " is out of range [0, "
            << nbatch_ << ")";
        throw py::value_error(msg.str());
      }
    }

    size_t sz_state = static_cast<size_t>(n) * nstate_;
    size_t sz_warmstart = static_cast<size_t>(n) * nv_;
    mjtNum* initial_state_ptr =
        required_array_ptr(initial_state, "initial_state", sz_state);
    mjtNum* initial_warmstart_ptr =
        optional_array_ptr(initial_warmstart, "initial_warmstart", sz_warmstart);

    // Per-k refresh flags derived from which fields the dict touches.
    std::vector<uint8_t> needs_refresh(n, 0);
    bool any_field_needs_refresh = false;

    if (randomization.has_value() && !randomization->empty()) {
      // Validate and apply patches. GIL is held here; keep this path lean.
      for (auto item : *randomization) {
        std::string name = item.first.cast<std::string>();
        const FieldSpec* spec = LookupField(name);
        if (!spec) {
          std::ostringstream msg;
          msg << "Unknown randomization field '" << name
              << "'. Supported: " << SupportedFieldsString();
          throw py::value_error(msg.str());
        }
        if (spec->needs_refresh) any_field_needs_refresh = true;

        // Coerce to contiguous float64.
        py::object payload_obj = np.attr("ascontiguousarray")(
            py::reinterpret_borrow<py::object>(item.second),
            py::arg("dtype") = np.attr("float64"));
        PyCArray payload = payload_obj.cast<PyCArray>();
        int field_size = FieldSize(spec->id, models_[0]);
        size_t expected = static_cast<size_t>(n) * field_size;
        py::buffer_info info = payload.request();
        if (static_cast<size_t>(info.size) != expected) {
          std::ostringstream msg;
          msg << "randomization['" << name << "'].size should be " << expected
              << " (n=" << n << " x " << field_size << "), got " << info.size;
          throw py::value_error(msg.str());
        }
        const mjtNum* src = static_cast<const mjtNum*>(info.ptr);

        // Apply per-env in C++ while the GIL is held.
        for (int k = 0; k < n; ++k) {
          int e = env_ids_ptr[k];
          WriteField(spec->id, models_[e], src + static_cast<size_t>(k) * field_size);
        }
        if (spec->needs_refresh) {
          for (int k = 0; k < n; ++k) needs_refresh[k] = 1;
        }
      }
    }

    const uint8_t* refresh_ptr =
        any_field_needs_refresh ? needs_refresh.data() : nullptr;

    mjtNum* state_ptr = static_cast<mjtNum*>(state_out.request().ptr);
    mjtNum* sensordata_ptr =
        static_cast<mjtNum*>(sensordata_out.request().ptr);

    {
      py::gil_scoped_release no_gil;
      if (nthread_ > 0 && n > 1) {
        int chunk = chunk_size.has_value()
                        ? *chunk_size
                        : std::max(1, n / (10 * nthread_));
        InterceptMjErrors(_unsafe_subset_reset_threaded)(
            models_, worker_data_, env_ids_ptr, n, initial_state_ptr,
            initial_warmstart_ptr, refresh_ptr, state_ptr, sensordata_ptr,
            skipsensor ? 1 : 0, pool_.get(), chunk);
      } else {
        InterceptMjErrors(_unsafe_subset_reset_range)(
            models_, worker_data_[0], env_ids_ptr, 0, n, initial_state_ptr,
            initial_warmstart_ptr, refresh_ptr, state_ptr, sensordata_ptr,
            skipsensor ? 1 : 0);
      }
    }

    return py::make_tuple(state_out, sensordata_out);
  }

  // -----------------------------------------------------------------
  // Introspection.
  // -----------------------------------------------------------------
  int nbatch() const { return nbatch_; }
  int nthread() const { return nthread_; }
  int nstate() const { return nstate_; }
  int nv() const { return nv_; }
  int nsensordata() const { return nsensordata_; }

  py::object get_model(int env_id) {
    ValidateEnvIdOrThrow(env_id, nbatch_);
    if (MjModelWrapper* wrapper = MjModelWrapper::FromRawPointer(models_[env_id])) {
      return py::cast(wrapper, py::return_value_policy::reference);
    }

    py::capsule owner(
        new py::object(py::cast(this, py::return_value_policy::reference)),
        nullptr, &KeepAlivePyObjectCapsuleDestructor);
    return py::cast(
        MjModelWrapper(models_[env_id], owner),
        py::return_value_policy::move);
  }

  py::list get_models(const PyIArray env_ids) {
    py::buffer_info index_info = env_ids.request();
    if (index_info.ndim != 1) {
      throw py::value_error("env_ids must be a 1-D array");
    }

    py::list out;
    const int* index_ptr = static_cast<const int*>(index_info.ptr);
    int n = static_cast<int>(index_info.shape[0]);
    for (int i = 0; i < n; ++i) {
      out.append(get_model(index_ptr[i]));
    }
    return out;
  }

  py::list get_all_models() {
    py::list out;
    for (int env_id = 0; env_id < nbatch_; ++env_id) {
      out.append(get_model(env_id));
    }
    return out;
  }

  // Return a read-only view of the given model's field as a 1-D numpy array.
  // Intended for tests. The returned array shares memory with the pool and
  // must not be mutated by the caller.
  py::array_t<mjtNum> get_field(int env_id, const std::string& name) {
    ValidateEnvIdOrThrow(env_id, nbatch_);
    const FieldSpec& spec = LookupFieldOrThrow(name);
    int sz = FieldSize(spec.id, models_[env_id]);
    // Copy into a fresh array to avoid lifetime entanglement with the pool.
    py::array_t<mjtNum> out(static_cast<py::ssize_t>(sz));
    CopyFieldOut(
        spec.id, models_[env_id], static_cast<mjtNum*>(out.request().ptr));
    return out;
  }

  py::array_t<mjtNum> get_field_indexed(int env_id, const std::string& name,
                                        const PyIArray indices) {
    ValidateEnvIdOrThrow(env_id, nbatch_);
    const FieldSpec& spec = LookupFieldOrThrow(name);
    py::buffer_info index_info = indices.request();
    if (index_info.ndim != 1) {
      throw py::value_error("indices must be a 1-D array");
    }

    int n = static_cast<int>(index_info.shape[0]);
    int width = FieldComponentWidth(spec.id);
    int index_count = FieldIndexCount(spec.id, models_[env_id]);
    std::vector<py::ssize_t> shape =
        width == 1 ? std::vector<py::ssize_t>{n}
                   : std::vector<py::ssize_t>{n, width};
    py::array_t<mjtNum> out(shape);
    mjtNum* dst = static_cast<mjtNum*>(out.request().ptr);
    const int* index_ptr = static_cast<const int*>(index_info.ptr);
    for (int i = 0; i < n; ++i) {
      ValidateFieldIndexOrThrow(index_ptr[i], index_count, name);
      CopyIndexedFieldOut(spec.id, models_[env_id], index_ptr[i],
                          dst + static_cast<size_t>(i) * width);
    }
    return out;
  }

  void set_field_indexed(int env_id, const std::string& name,
                         const PyIArray indices, const PyCArray value) {
    ValidateEnvIdOrThrow(env_id, nbatch_);
    const FieldSpec& spec = LookupFieldOrThrow(name);
    py::buffer_info index_info = indices.request();
    if (index_info.ndim != 1) {
      throw py::value_error("indices must be a 1-D array");
    }

    int n = static_cast<int>(index_info.shape[0]);
    int width = FieldComponentWidth(spec.id);
    size_t expected = static_cast<size_t>(n) * width;
    py::buffer_info value_info = value.request();
    if (static_cast<size_t>(value_info.size) != expected) {
      std::ostringstream msg;
      msg << "value.size should be " << expected << ", got " << value_info.size;
      throw py::value_error(msg.str());
    }

    int index_count = FieldIndexCount(spec.id, models_[env_id]);
    const int* index_ptr = static_cast<const int*>(index_info.ptr);
    const mjtNum* src = static_cast<const mjtNum*>(value_info.ptr);
    for (int i = 0; i < n; ++i) {
      ValidateFieldIndexOrThrow(index_ptr[i], index_count, name);
      WriteIndexedField(spec.id, models_[env_id], index_ptr[i],
                        src + static_cast<size_t>(i) * width);
    }
    if (spec.needs_refresh && n > 0) {
      InterceptMjErrors(mj_setConst)(models_[env_id], worker_data_[0]);
    }
  }

 private:
  int nbatch_;
  int nthread_;
  int nstate_ = 0;
  int nv_ = 0;
  int nsensordata_ = 0;
  std::vector<raw::MjModel*> models_;
  std::vector<raw::MjData*> worker_data_;
  std::shared_ptr<ThreadPool> pool_;
};

PYBIND11_MODULE(_batch_env, pymodule) {
  py::class_<BatchEnvPool>(pymodule, "BatchEnvPool")
      .def(py::init([](py::object base_model, int nbatch, int nthread) {
             return std::make_unique<BatchEnvPool>(base_model, nbatch, nthread);
           }),
           py::kw_only(), py::arg("model"), py::arg("nbatch"),
           py::arg("nthread"))
      .def("step", &BatchEnvPool::step, py::kw_only(), py::arg("nstep"),
           py::arg("control_spec"), py::arg("state0"),
           py::arg("warmstart0") = py::none(),
           py::arg("control") = py::none(),
           py::arg("chunk_size") = py::none(),
           py::arg("return_sensor") = false,
           py::arg("post_step_forward_sensor") = false)
      .def("forward", &BatchEnvPool::forward, py::kw_only(),
           py::arg("state0"), py::arg("warmstart0") = py::none(),
           py::arg("skipsensor") = false, py::arg("chunk_size") = py::none())
      .def("reset", &BatchEnvPool::reset, py::kw_only(),
           py::arg("env_ids"), py::arg("initial_state"),
           py::arg("randomization") = py::none(),
           py::arg("initial_warmstart") = py::none(),
           py::arg("skipsensor") = false, py::arg("chunk_size") = py::none())
      .def("compute_site_jacobians", &BatchEnvPool::compute_site_jacobians,
           py::kw_only(), py::arg("state0"), py::arg("site_ids"),
           py::arg("jacp") = true, py::arg("jacr") = false,
           py::arg("warmstart0") = py::none(),
           py::arg("chunk_size") = py::none())
      .def("sample_hfield_height", &BatchEnvPool::sample_hfield_height,
           py::kw_only(), py::arg("state0"), py::arg("hfield_geom_id"),
           py::arg("offsets"), py::arg("frame_body_id"),
           py::arg("alignment") = "yaw", py::arg("output") = "height",
           py::arg("chunk_size") = py::none())
      .def("get_model", &BatchEnvPool::get_model, py::arg("env_id"))
      .def("get_models", &BatchEnvPool::get_models, py::arg("env_ids"))
      .def("get_all_models", &BatchEnvPool::get_all_models)
      .def("get_field", &BatchEnvPool::get_field, py::arg("env_id"),
           py::arg("name"))
      .def("get_field_indexed", &BatchEnvPool::get_field_indexed,
           py::arg("env_id"), py::arg("name"), py::arg("indices"))
      .def("set_field_indexed", &BatchEnvPool::set_field_indexed,
           py::arg("env_id"), py::arg("name"), py::arg("indices"),
           py::arg("value"))
      .def_property_readonly("nbatch", &BatchEnvPool::nbatch)
      .def_property_readonly("nthread", &BatchEnvPool::nthread)
      .def_property_readonly("nstate", &BatchEnvPool::nstate)
      .def_property_readonly("nv", &BatchEnvPool::nv)
      .def_property_readonly("nsensordata", &BatchEnvPool::nsensordata);

  pymodule.attr("SUPPORTED_FIELDS") = []() {
    py::list out;
    for (const FieldSpec& spec : kFieldSpecs) {
      out.append(py::str(spec.name));
    }
    return out;
  }();
}

}  // namespace

}  // namespace mujoco::python
