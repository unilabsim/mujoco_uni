// Copyright 2022 DeepMind Technologies Limited
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
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

void _unsafe_mlx_step_range(std::vector<const mjModel*>& m, mjData* d, int start_batch,
                            int end_batch, int nstep, unsigned int control_spec,
                            const mjtNum* state0, const mjtNum* warmstart0,
                            const mjtNum* control, mjtNum* state, mjtNum* sensordata) {
  size_t nstate = static_cast<size_t>(mj_stateSize(m[0], mjSTATE_FULLPHYSICS));
  size_t ncontrol = static_cast<size_t>(mj_stateSize(m[0], control_spec));
  size_t nv = static_cast<size_t>(m[0]->nv);
  int nbody = m[0]->nbody, neq = m[0]->neq;
  size_t nsensordata = static_cast<size_t>(m[0]->nsensordata);

  if (!(control_spec & mjSTATE_CTRL)) mju_zero(d->ctrl, m[0]->nu);
  if (!(control_spec & mjSTATE_QFRC_APPLIED)) mju_zero(d->qfrc_applied, nv);
  if (!(control_spec & mjSTATE_XFRC_APPLIED)) mju_zero(d->xfrc_applied, 6 * nbody);

  for (size_t r = start_batch; r < end_batch; r++) {
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
      for (int i = 0; i < neq; i++) d->eq_active[i] = m[r]->eq_active0[i];
    }

    mj_setState(m[r], d, state0 + r * nstate, mjSTATE_FULLPHYSICS);
    if (warmstart0) mju_copy(d->qacc_warmstart, warmstart0 + r * nv, nv);
    else mju_zero(d->qacc_warmstart, nv);
    for (int i = 0; i < mjNWARNING; i++) d->warning[i].number = 0;

    for (size_t t = 0; t < nstep; t++) {
      bool nwarning = false;
      for (int i = 0; i < mjNWARNING; i++) {
        if (d->warning[i].number) {
          nwarning = true;
          break;
        }
      }
      if (nwarning) {
        for (; t < nstep; t++) {
          size_t step = r * static_cast<size_t>(nstep) + t;
          if (state) mj_getState(m[r], d, state + step * nstate, mjSTATE_FULLPHYSICS);
          if (sensordata) mju_copy(sensordata + step * nsensordata, d->sensordata, nsensordata);
        }
        break;
      }

      size_t step = r * static_cast<size_t>(nstep) + t;
      if (control) mj_setState(m[r], d, control + step * ncontrol, control_spec);
      mj_step(m[r], d);
      if (state) mj_getState(m[r], d, state + step * nstate, mjSTATE_FULLPHYSICS);
      if (sensordata) mju_copy(sensordata + step * nsensordata, d->sensordata, nsensordata);
    }
  }
}

void _unsafe_mlx_step_threaded(std::vector<const mjModel*>& m, std::vector<mjData*>& d,
                               int nbatch, int nstep, unsigned int control_spec,
                               const mjtNum* state0, const mjtNum* warmstart0,
                               const mjtNum* control, mjtNum* state, mjtNum* sensordata,
                               ThreadPool* pool, int chunk_size) {
  int nfulljobs = nbatch / chunk_size;
  int chunk_remainder = nbatch % chunk_size;
  int njobs = chunk_remainder > 0 ? nfulljobs + 1 : nfulljobs;
  pool->ResetCount();
  for (int j = 0; j < nfulljobs; j++) {
    auto task = [=, &m, &d](void) {
      int id = pool->WorkerId();
      _unsafe_mlx_step_range(m, d[id], j * chunk_size, (j + 1) * chunk_size, nstep,
                             control_spec, state0, warmstart0, control, state, sensordata);
    };
    pool->Schedule(task);
  }
  if (chunk_remainder > 0) {
    auto task = [=, &m, &d](void) {
      _unsafe_mlx_step_range(m, d[pool->WorkerId()], nfulljobs * chunk_size,
                             nfulljobs * chunk_size + chunk_remainder, nstep, control_spec,
                             state0, warmstart0, control, state, sensordata);
    };
    pool->Schedule(task);
  }
  pool->WaitCount(njobs);
}

PyCArray to_numpy_f64(py::handle arg, const char* name) {
  if (arg.is_none()) {
    std::ostringstream msg;
    msg << name << " cannot be None";
    throw py::value_error(msg.str());
  }
  py::module_ np = py::module_::import("numpy");
  py::object arr = np.attr("ascontiguousarray")(np.attr("asarray")(arg),
                                                 py::arg("dtype") = np.attr("float64"));
  return arr.cast<PyCArray>();
}

std::optional<PyCArray> to_optional_numpy_f64(const std::optional<py::object>& arg,
                                               const char* name) {
  if (!arg.has_value() || arg->is_none()) return std::nullopt;
  return to_numpy_f64(*arg, name);
}

mjtNum* get_array_ptr(const std::optional<PyCArray>& arg, const char* name, int nbatch,
                      int nstep, int dim) {
  if (!arg.has_value()) return nullptr;
  py::buffer_info info = arg->request();
  size_t expected = static_cast<size_t>(nbatch) * static_cast<size_t>(nstep) *
                    static_cast<size_t>(dim);
  if (info.size != expected) {
    std::ostringstream msg;
    msg << name << ".size should be " << expected << ", got " << info.size;
    throw py::value_error(msg.str());
  }
  return static_cast<mjtNum*>(info.ptr);
}

class MlxStepRunner {
 public:
  explicit MlxStepRunner(int nthread) : nthread_(nthread) {
    if (nthread_ > 0) pool_ = std::make_shared<ThreadPool>(nthread_);
  }

  py::tuple step(py::list m, py::list d, int nstep, unsigned int control_spec,
                 py::object state0, std::optional<py::object> warmstart0,
                 std::optional<py::object> control, std::optional<int> chunk_size,
                 std::optional<py::object> out_dtype) {
    PyCArray state0_np = to_numpy_f64(state0, "state0");
    std::optional<PyCArray> warmstart0_np = to_optional_numpy_f64(warmstart0, "warmstart0");
    std::optional<PyCArray> control_np = to_optional_numpy_f64(control, "control");

    int nbatch = state0_np.shape(0);
    std::vector<const raw::MjModel*> model_ptrs(nbatch);
    for (int i = 0; i < nbatch; i++) model_ptrs[i] = m[i].cast<const MjModelWrapper*>()->get();

    if (py::len(d) == 0) throw py::value_error("The list of data instances is empty");
    if (nthread_ == 0 && py::len(d) > 1) {
      throw py::value_error("More than one data instance passed but step runner is configured to run on main thread");
    }
    if (nthread_ > 0 && nthread_ != py::len(d)) {
      std::ostringstream msg;
      msg << "Length of data: " << py::len(d) << " not equal to nthread: " << nthread_;
      throw py::value_error(msg.str());
    }

    std::vector<raw::MjData*> data_ptrs(py::len(d));
    for (int t = 0; t < py::len(d); t++) data_ptrs[t] = d[t].cast<MjDataWrapper*>()->get();

    int nstate = mj_stateSize(model_ptrs[0], mjSTATE_FULLPHYSICS);
    int ncontrol = mj_stateSize(model_ptrs[0], control_spec);
    int nv = model_ptrs[0]->nv;
    int nsensordata = model_ptrs[0]->nsensordata;

    std::optional<PyCArray> state0_opt = state0_np;
    mjtNum* state0_ptr = get_array_ptr(state0_opt, "state0", nbatch, 1, nstate);
    mjtNum* warmstart0_ptr = get_array_ptr(warmstart0_np, "warmstart0", nbatch, 1, nv);
    mjtNum* control_ptr = get_array_ptr(control_np, "control", nbatch, nstep, ncontrol);

    PyCArray state_np({nbatch, nstep, nstate});
    PyCArray sensordata_np({nbatch, nstep, nsensordata});
    std::optional<PyCArray> state_opt = state_np;
    std::optional<PyCArray> sensordata_opt = sensordata_np;
    mjtNum* state_ptr = get_array_ptr(state_opt, "state", nbatch, nstep, nstate);
    mjtNum* sensordata_ptr = get_array_ptr(sensordata_opt, "sensordata", nbatch, nstep, nsensordata);

    if (nstep > 0) {
      py::gil_scoped_release no_gil;
      if (nthread_ > 0 && nbatch > 1) {
        int chunk = chunk_size.has_value() ? *chunk_size : std::max(1, nbatch / (10 * nthread_));
        InterceptMjErrors(_unsafe_mlx_step_threaded)(
            model_ptrs, data_ptrs, nbatch, nstep, control_spec, state0_ptr, warmstart0_ptr,
            control_ptr, state_ptr, sensordata_ptr, pool_.get(), chunk);
      } else {
        InterceptMjErrors(_unsafe_mlx_step_range)(
            model_ptrs, data_ptrs[0], 0, nbatch, nstep, control_spec, state0_ptr, warmstart0_ptr,
            control_ptr, state_ptr, sensordata_ptr);
      }
    }

    py::module_ mlx = py::module_::import("mlx.core");
    py::object dtype = out_dtype.has_value() ? *out_dtype : mlx.attr("float32");
    py::object state_mx = mlx.attr("array")(state_np, py::arg("dtype") = dtype);
    py::object sensor_mx = mlx.attr("array")(sensordata_np, py::arg("dtype") = dtype);
    return py::make_tuple(state_mx, sensor_mx);
  }

 private:
  int nthread_;
  std::shared_ptr<ThreadPool> pool_;
};

PYBIND11_MODULE(_mlx_step, pymodule) {
  py::class_<MlxStepRunner>(pymodule, "MlxStepRunner")
      .def(py::init([](int nthread) { return std::make_unique<MlxStepRunner>(nthread); }),
           py::kw_only(), py::arg("nthread"))
      .def("step", &MlxStepRunner::step, py::arg("model"), py::arg("data"), py::arg("nstep"),
           py::arg("control_spec"), py::arg("state0"), py::arg("warmstart0") = py::none(),
           py::arg("control") = py::none(), py::arg("chunk_size") = py::none(),
           py::arg("out_dtype") = py::none());
}

}  // namespace
}  // namespace mujoco::python
