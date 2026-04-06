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

// ===================================================================
// Field registry — supported reset-lifecycle randomization fields.
// ===================================================================

enum class FieldId {
  kBodyMass,
  kBodyIpos,
  kGeomFriction,
  kDofArmature,
  kDofDamping,
};

struct FieldSpec {
  const char* name;
  FieldId id;
  bool needs_refresh;  // needs mj_setConst after writing
};

constexpr FieldSpec kFieldSpecs[] = {
    {"body_mass",     FieldId::kBodyMass,     true},
    {"body_ipos",     FieldId::kBodyIpos,     true},
    {"dof_armature",  FieldId::kDofArmature,  true},
    {"geom_friction", FieldId::kGeomFriction, false},
    {"dof_damping",   FieldId::kDofDamping,   false},
};

// Element count of the field for a given model.
int FieldSize(FieldId id, const raw::MjModel* m) {
  switch (id) {
    case FieldId::kBodyMass:     return m->nbody;
    case FieldId::kBodyIpos:     return 3 * m->nbody;
    case FieldId::kDofArmature:  return m->nv;
    case FieldId::kGeomFriction: return 3 * m->ngeom;
    case FieldId::kDofDamping:   return m->nv;
  }
  return 0;
}

// Pointer to the field storage in m.
mjtNum* FieldPtr(FieldId id, raw::MjModel* m) {
  switch (id) {
    case FieldId::kBodyMass:     return m->body_mass;
    case FieldId::kBodyIpos:     return m->body_ipos;
    case FieldId::kDofArmature:  return m->dof_armature;
    case FieldId::kGeomFriction: return m->geom_friction;
    case FieldId::kDofDamping:   return m->dof_damping;
  }
  return nullptr;
}

const FieldSpec* LookupField(const std::string& name) {
  for (const FieldSpec& spec : kFieldSpecs) {
    if (name == spec.name) return &spec;
  }
  return nullptr;
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

// ===================================================================
// Kernels — mirror rollout.cc / batch_forward_bind.cc, operating over
// the pool's owned models/data.
// ===================================================================

// Full rollout of envs [start_roll, end_roll) using stepping.
// Copy of rollout.cc::_unsafe_rollout; kept local to this TU.
void _unsafe_full_rollout(const std::vector<const raw::MjModel*>& m,
                          raw::MjData* d, int start_roll, int end_roll,
                          int nstep, unsigned int control_spec,
                          const mjtNum* state0, const mjtNum* warmstart0,
                          const mjtNum* control, mjtNum* state,
                          mjtNum* sensordata) {
  size_t nstate = static_cast<size_t>(mj_stateSize(m[0], mjSTATE_FULLPHYSICS));
  size_t ncontrol = static_cast<size_t>(mj_stateSize(m[0], control_spec));
  size_t nv = static_cast<size_t>(m[0]->nv);
  int nbody = m[0]->nbody, neq = m[0]->neq;
  size_t nsensordata = static_cast<size_t>(m[0]->nsensordata);

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
      if (nwarning) {
        for (; t < static_cast<size_t>(nstep); t++) {
          size_t step = r * static_cast<size_t>(nstep) + t;
          if (state) {
            mj_getState(m[r], d, state + step * nstate, mjSTATE_FULLPHYSICS);
          }
          if (sensordata) {
            mju_copy(sensordata + step * nsensordata, d->sensordata,
                     nsensordata);
          }
        }
        break;
      }

      size_t step = r * static_cast<size_t>(nstep) + t;

      if (control) {
        mj_setState(m[r], d, control + step * ncontrol, control_spec);
      }

      mj_step(m[r], d);

      if (state) {
        mj_getState(m[r], d, state + step * nstate, mjSTATE_FULLPHYSICS);
      }
      if (sensordata) {
        mju_copy(sensordata + step * nsensordata, d->sensordata, nsensordata);
      }
    }
  }
}

void _unsafe_full_rollout_threaded(
    const std::vector<const raw::MjModel*>& m, std::vector<raw::MjData*>& d,
    int nbatch, int nstep, unsigned int control_spec, const mjtNum* state0,
    const mjtNum* warmstart0, const mjtNum* control, mjtNum* state,
    mjtNum* sensordata, ThreadPool* pool, int chunk_size) {
  int nfulljobs = nbatch / chunk_size;
  int chunk_remainder = nbatch % chunk_size;
  int njobs = (chunk_remainder > 0) ? nfulljobs + 1 : nfulljobs;

  pool->ResetCount();

  for (int j = 0; j < nfulljobs; j++) {
    auto task = [=, &m, &d](void) {
      int id = pool->WorkerId();
      _unsafe_full_rollout(m, d[id], j * chunk_size, (j + 1) * chunk_size,
                           nstep, control_spec, state0, warmstart0, control,
                           state, sensordata);
    };
    pool->Schedule(task);
  }
  if (chunk_remainder > 0) {
    auto task = [=, &m, &d](void) {
      _unsafe_full_rollout(m, d[pool->WorkerId()], nfulljobs * chunk_size,
                           nfulljobs * chunk_size + chunk_remainder, nstep,
                           control_spec, state0, warmstart0, control, state,
                           sensordata);
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

class DomainRandomizedRolloutPool {
 public:
  DomainRandomizedRolloutPool(py::object base_model_obj, int nbatch, int nthread)
      : nbatch_(nbatch), nthread_(nthread) {
    if (nbatch_ <= 0) {
      throw py::value_error("nbatch must be positive");
    }
    if (nthread_ < 0) {
      throw py::value_error("nthread must be >= 0");
    }

    const raw::MjModel* base =
        base_model_obj.cast<const MjModelWrapper*>()->get();

    models_.resize(nbatch_, nullptr);
    for (int i = 0; i < nbatch_; ++i) {
      models_[i] = mj_copyModel(nullptr, base);
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

  ~DomainRandomizedRolloutPool() {
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
  // Full-batch rollout.
  // -----------------------------------------------------------------
  void rollout(int nstep, unsigned int control_spec, const PyCArray state0,
               std::optional<const PyCArray> warmstart0,
               std::optional<const PyCArray> control,
               std::optional<const PyCArray> state,
               std::optional<const PyCArray> sensordata,
               std::optional<int> chunk_size) {
    if (nstep < 1) {
      return;
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
    size_t sz_state =
        static_cast<size_t>(nbatch_) * nstep * nstate_;
    size_t sz_sensordata =
        static_cast<size_t>(nbatch_) * nstep * nsensordata_;

    std::optional<PyCArray> state0_opt = state0;
    mjtNum* state0_ptr = optional_array_ptr(state0_opt, "state0", sz_state0);
    mjtNum* warmstart0_ptr =
        optional_array_ptr(warmstart0, "warmstart0", sz_warmstart);
    mjtNum* control_ptr = optional_array_ptr(control, "control", sz_control);
    mjtNum* state_ptr = optional_array_ptr(state, "state", sz_state);
    mjtNum* sensordata_ptr =
        optional_array_ptr(sensordata, "sensordata", sz_sensordata);

    std::vector<const raw::MjModel*> cmodels(nbatch_);
    for (int i = 0; i < nbatch_; ++i) cmodels[i] = models_[i];

    {
      py::gil_scoped_release no_gil;
      if (nthread_ > 0 && nbatch_ > 1) {
        int chunk = chunk_size.has_value()
                        ? *chunk_size
                        : std::max(1, nbatch_ / (10 * nthread_));
        InterceptMjErrors(_unsafe_full_rollout_threaded)(
            cmodels, worker_data_, nbatch_, nstep, control_spec, state0_ptr,
            warmstart0_ptr, control_ptr, state_ptr, sensordata_ptr, pool_.get(),
            chunk);
      } else {
        InterceptMjErrors(_unsafe_full_rollout)(
            cmodels, worker_data_[0], 0, nbatch_, nstep, control_spec,
            state0_ptr, warmstart0_ptr, control_ptr, state_ptr, sensordata_ptr);
      }
    }
  }

  // -----------------------------------------------------------------
  // Sparse reset with optional per-env model patching.
  //
  // randomization: dict[str, ndarray] where each ndarray has leading dim
  //                equal to len(env_ids) and trailing size equal to the
  //                field's element count for the base model.
  // -----------------------------------------------------------------
  py::tuple reset_subset(py::array env_ids_arr, const PyCArray initial_state,
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

        // Apply per-env in C++ (GIL held — this loop is strictly memcpy).
        for (int k = 0; k < n; ++k) {
          int e = env_ids_ptr[k];
          mjtNum* dst = FieldPtr(spec->id, models_[e]);
          mju_copy(dst, src + static_cast<size_t>(k) * field_size, field_size);
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

  // Return a read-only view of the given model's field as a 1-D numpy array.
  // Intended for tests. The returned array shares memory with the pool and
  // must not be mutated by the caller.
  py::array_t<mjtNum> get_field(int env_id, const std::string& name) {
    if (env_id < 0 || env_id >= nbatch_) {
      throw py::value_error("env_id out of range");
    }
    const FieldSpec* spec = LookupField(name);
    if (!spec) {
      std::ostringstream msg;
      msg << "Unknown field '" << name
          << "'. Supported: " << SupportedFieldsString();
      throw py::value_error(msg.str());
    }
    int sz = FieldSize(spec->id, models_[env_id]);
    mjtNum* ptr = FieldPtr(spec->id, models_[env_id]);
    // Copy into a fresh array to avoid lifetime entanglement with the pool.
    py::array_t<mjtNum> out(static_cast<py::ssize_t>(sz));
    mju_copy(static_cast<mjtNum*>(out.request().ptr), ptr, sz);
    return out;
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

PYBIND11_MODULE(_dr_rollout, pymodule) {
  py::class_<DomainRandomizedRolloutPool>(pymodule, "DomainRandomizedRolloutPool")
      .def(py::init([](py::object base_model, int nbatch, int nthread) {
             return std::make_unique<DomainRandomizedRolloutPool>(
                 base_model, nbatch, nthread);
           }),
           py::kw_only(), py::arg("model"), py::arg("nbatch"),
           py::arg("nthread"))
      .def("rollout", &DomainRandomizedRolloutPool::rollout,
           py::kw_only(), py::arg("nstep"), py::arg("control_spec"),
           py::arg("state0"), py::arg("warmstart0") = py::none(),
           py::arg("control") = py::none(), py::arg("state") = py::none(),
           py::arg("sensordata") = py::none(),
           py::arg("chunk_size") = py::none())
      .def("reset_subset", &DomainRandomizedRolloutPool::reset_subset,
           py::kw_only(), py::arg("env_ids"), py::arg("initial_state"),
           py::arg("randomization") = py::none(),
           py::arg("initial_warmstart") = py::none(),
           py::arg("skipsensor") = false, py::arg("chunk_size") = py::none())
      .def("get_field", &DomainRandomizedRolloutPool::get_field,
           py::arg("env_id"), py::arg("name"))
      .def_property_readonly("nbatch", &DomainRandomizedRolloutPool::nbatch)
      .def_property_readonly("nthread", &DomainRandomizedRolloutPool::nthread)
      .def_property_readonly("nstate", &DomainRandomizedRolloutPool::nstate)
      .def_property_readonly("nv", &DomainRandomizedRolloutPool::nv)
      .def_property_readonly("nsensordata",
                             &DomainRandomizedRolloutPool::nsensordata);

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
