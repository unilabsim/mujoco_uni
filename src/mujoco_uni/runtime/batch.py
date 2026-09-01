"""Batched environment pool backed by a persistent per-env mjModel pool.

This module wraps the pybind-backed ``_batch_env`` C++ module. The pool
owns:
  * ``nbatch`` ``mjModel`` instances (cloned from one caller-supplied
    model, or from a caller-supplied compatible model sequence, via
    ``mj_copyModel``),
  * per-thread ``mjData`` workers,
  * an internal thread pool.

It exposes three execution primitives:

  * :meth:`BatchEnvPool.step` — multi-step ``mj_step`` over the full
    env pool. Returns only the **final** state, and can optionally also
    return the final-step sensordata
    (``(nbatch, nstate)`` / ``(nbatch, nsensordata)``), not trajectories.

  * :meth:`BatchEnvPool.forward` — single ``mj_forward`` over all envs.
    Returns only ``sensordata``.

  * :meth:`BatchEnvPool.reset` — fused sparse reset over a subset of
    envs with optional per-env model field patching and selective
    ``mj_setConst`` refresh.

  * :meth:`BatchEnvPool.sample_hfield_height` — MuJoCo hfield-only
    yaw/body/world aligned bilinear terrain sampling over the full env pool.

Supported randomization fields are listed in ``SUPPORTED_FIELDS``.
"""

from __future__ import annotations

import numbers
import os
import re
import sys
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from mujoco_uni.metadata import (
  MUJOCO_MAX_VERSION_EXCLUSIVE,
  MUJOCO_MIN_VERSION,
  MUJOCO_VERSION_SPEC,
  __version__,
)
from mujoco_uni.mujoco_runtime import api as mujoco


def _parse_mujoco_version(version: str) -> tuple[int, int, int]:
  match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\.|$)", version)
  if match is None:
    raise ImportError(f"Unsupported MuJoCo version string: {version!r}")
  return tuple(int(match.group(i)) for i in range(1, 4))


def _check_mujoco_python_compatibility() -> str:
  version = getattr(mujoco, "__version__", "")
  parsed = _parse_mujoco_version(version)
  min_version = _parse_mujoco_version(MUJOCO_MIN_VERSION)
  max_version = _parse_mujoco_version(MUJOCO_MAX_VERSION_EXCLUSIVE)
  if parsed < min_version or parsed >= max_version:
    raise ImportError(
        f"mujoco_uni {__version__} supports official mujoco{MUJOCO_VERSION_SPEC}; "
        f"found mujoco {version!r}"
    )
  if not hasattr(mujoco.MjModel, "_address"):
    raise ImportError("mujoco.MjModel._address is required by mujoco_uni")
  if not hasattr(mujoco.MjModel, "_from_model_ptr"):
    raise ImportError("mujoco.MjModel._from_model_ptr is required by mujoco_uni")
  return version


def _check_native_build_compatibility(native, runtime_version: str) -> None:
  build_version = getattr(native, "MUJOCO_BUILD_VERSION", "unknown")
  if build_version == "unknown":
    raise ImportError(
        "MuJoCoUni native batch extension does not expose its MuJoCo build "
        "version; rebuild the extension"
    )
  if _parse_mujoco_version(str(build_version)) != _parse_mujoco_version(runtime_version):
    raise ImportError(
        f"MuJoCoUni native batch extension was built against mujoco "
        f"{build_version!r}, but loaded mujoco is {runtime_version!r}. "
        "Rebuild mujoco_uni inside the selected MuJoCo environment."
    )


_runtime_mujoco_version = _check_mujoco_python_compatibility()

from mujoco_uni.compiled import require_native

_native = require_native()
_check_native_build_compatibility(_native, _runtime_mujoco_version)


SUPPORTED_FIELDS = tuple(_native.SUPPORTED_FIELDS)
_FIELD_COMPONENT_WIDTHS = {
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


def _normalize_indices(indices) -> tuple[np.ndarray, bool]:
  if isinstance(indices, (bool, np.bool_)):
    raise TypeError("indices must be an int or a 1-D sequence of ints")
  if isinstance(indices, numbers.Integral):
    return np.asarray([int(indices)], dtype=np.int32), True
  if isinstance(indices, (str, bytes)):
    raise TypeError("indices must be an int or a 1-D sequence of ints")

  arr = np.asarray(indices)
  if arr.ndim != 1:
    raise ValueError(f"indices must be 1-D, got shape {arr.shape}")
  if arr.dtype.kind not in ("i", "u"):
    raise TypeError("indices must contain integers")
  return np.ascontiguousarray(arr, dtype=np.int32), False


def _normalize_scalar_int(name: str, value) -> int:
  if isinstance(value, (bool, np.bool_)):
    raise TypeError(f"{name} must be an integer id")
  if not isinstance(value, numbers.Integral):
    raise TypeError(f"{name} must be an integer id")
  return int(value)


def _normalize_cpu_ids(cpu_ids, nthread: Optional[int]) -> tuple[list[int], int]:
  """Validate ``cpu_ids`` and resolve the effective worker count.

  ``cpu_ids[i]`` pins worker thread ``i`` to one CPU, so the mapping is
  1:1 with ``nthread``. When ``nthread`` is ``None`` it is inferred from
  ``len(cpu_ids)``.
  """
  if sys.platform != "linux":
    raise ValueError("cpu_ids pinning is only supported on Linux")
  if isinstance(cpu_ids, (str, bytes)):
    raise TypeError("cpu_ids must be a sequence of integer CPU ids")
  ids = [_normalize_scalar_int("cpu_ids entry", value) for value in cpu_ids]
  if not ids:
    raise ValueError("cpu_ids must be non-empty")
  if any(cpu_id < 0 for cpu_id in ids):
    raise ValueError("cpu_ids entries must be >= 0")
  if len(set(ids)) != len(ids):
    raise ValueError("cpu_ids entries must be unique")
  available = os.sched_getaffinity(0)
  unavailable = [cpu_id for cpu_id in ids if cpu_id not in available]
  if unavailable:
    raise ValueError(
        f"cpu_ids {unavailable} are not available to this process"
    )
  if nthread is None:
    return ids, len(ids)
  if int(nthread) != len(ids):
    raise ValueError(
        f"cpu_ids length ({len(ids)}) must equal nthread ({int(nthread)})"
    )
  return ids, int(nthread)


def _normalize_indexed_value(name: str, scalar_index: bool, nindex: int, value):
  width = _FIELD_COMPONENT_WIDTHS[name]
  arr = np.asarray(value, dtype=np.float64)
  expected_shape = (
      ()
      if scalar_index and width == 1
      else (width,)
      if scalar_index
      else (nindex,)
      if width == 1
      else (nindex, width)
  )
  if arr.shape != expected_shape:
    raise ValueError(
        f"value for field '{name}' must have shape {expected_shape}, "
        f"got {arr.shape}"
    )
  return np.ascontiguousarray(arr.reshape(-1), dtype=np.float64)


class BatchEnvPool:
  """Persistent per-environment model pool with step / forward / reset."""

  def __init__(
      self,
      model: Union[mujoco.MjModel, Sequence[mujoco.MjModel]],
      *,
      nbatch: int,
      nthread: Optional[int] = None,
      cpu_ids: Optional[Sequence[int]] = None,
  ):
    """Construct a batch pool from one model or a compatible model sequence.

    Args:
      model: A single ``MjModel`` to clone across the pool, or a compatible
        sequence of ``MjModel`` instances with length ``1`` or ``nbatch``.
      nbatch: Number of environments in the pool.
      nthread: Number of worker threads. ``None`` means ``0``.
      cpu_ids: Optional explicit worker CPU affinity (Linux only).
        ``cpu_ids[i]`` pins worker thread ``i`` to one CPU, so its length
        must equal ``nthread``; when ``nthread`` is ``None`` it is inferred
        from ``len(cpu_ids)``. Requires threaded mode (``nthread >= 1``).
        ``None`` keeps the default OS scheduling behavior.
    """
    if nbatch <= 0:
      raise ValueError("nbatch must be positive")
    if not isinstance(model, mujoco.MjModel):
      model = list(model)
    native_cpu_ids: list[int] = []
    if cpu_ids is not None:
      native_cpu_ids, nthread = _normalize_cpu_ids(cpu_ids, nthread)
    self._nthread = 0 if nthread is None else int(nthread)
    self._pool = _native.BatchEnvPool(
        model=model,
        nbatch=int(nbatch),
        nthread=self._nthread,
        cpu_ids=native_cpu_ids,
    )

  def __enter__(self) -> "BatchEnvPool":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()

  def close(self) -> None:
    if self._pool is not None:
      del self._pool
      self._pool = None

  # -- introspection --------------------------------------------------
  @property
  def nbatch(self) -> int:
    return self._pool.nbatch

  @property
  def nthread(self) -> int:
    return self._pool.nthread

  @property
  def nstate(self) -> int:
    return self._pool.nstate

  @property
  def nv(self) -> int:
    return self._pool.nv

  @property
  def nsensordata(self) -> int:
    return self._pool.nsensordata

  @property
  def was_autoreset(self) -> np.ndarray:
    """Boolean mask for environments autoreset during the last step call."""
    if self._pool is None:
      raise RuntimeError("was_autoreset requested after pool close")
    return np.asarray(self._pool.was_autoreset, dtype=np.bool_)

  @property
  def cpu_ids(self) -> Optional[tuple[int, ...]]:
    """Configured worker→CPU mapping, or ``None`` when affinity is unset."""
    if self._pool is None:
      raise RuntimeError("cpu_ids requested after pool close")
    ids = tuple(self._pool.cpu_ids)
    return ids or None

  def worker_cpu_ids(self) -> tuple[int, ...]:
    """Per-worker CPU observed at pool startup, after pinning.

    Entry ``i`` is the CPU worker ``i`` was actually running on once its
    affinity was applied. Empty when ``cpu_ids`` was not configured.
    """
    if self._pool is None:
      raise RuntimeError("worker_cpu_ids requested after pool close")
    return tuple(self._pool.worker_cpu_ids())

  def get_all_models(self) -> list[mujoco.MjModel]:
    """Return pool-owned models without copying.

    Returned model objects are valid while the pool remains open.
    """
    if self._pool is None:
      raise RuntimeError("get_all_models requested after pool close")
    return self._pool.get_all_models()

  def get_model(
      self, env_ids: int | Sequence[int]
  ) -> mujoco.MjModel | list[mujoco.MjModel]:
    """Return one or more pool-owned models without copying.

    Returned model objects are valid while the pool remains open.
    """
    if self._pool is None:
      raise RuntimeError("get_model requested after pool close")
    env_ids_arr, scalar_index = _normalize_indices(env_ids)
    if scalar_index:
      return self._pool.get_model(int(env_ids_arr[0]))
    return self._pool.get_models(env_ids_arr)

  # -- step -----------------------------------------------------------
  def step(
      self,
      initial_state,
      *,
      nstep: int,
      control_spec: int = int(mujoco.mjtState.mjSTATE_CTRL),
      control=None,
      initial_warmstart=None,
      chunk_size: Optional[int] = None,
      return_sensor: bool = False,
      post_step_forward_sensor: bool = False,
      control_callback=None,
      callback_sensordata: bool = True,
  ):
    """Run ``nstep`` of ``mj_step`` on every environment.

    Returns only the **final** state after all steps by default. When
    ``return_sensor`` is true, returns ``(state, sensordata)`` where
    ``sensordata`` is only the final step's sensor data, not a trajectory.

    Args:
      initial_state: ``(nbatch, nstate)`` full-physics initial states.
      nstep: number of steps.
      control_spec: MuJoCo ``mjtState`` flags for control.
      control: ``(nbatch, nstep, ncontrol)`` control trajectories, optional.
        Mutually exclusive with ``control_callback``.
      initial_warmstart: ``(nbatch, nv)`` qacc_warmstart, optional.
      chunk_size: thread-pool chunk size, optional.
      return_sensor: return final-step sensordata together with state.
      post_step_forward_sensor: if returning sensors, run one ``mj_forward`` on
        the final state before copying sensordata. This matches the previous
        ``step`` followed by ``forward`` refresh behavior. When false, the
        sensordata left by the final ``mj_step`` is returned directly. With
        ``control_callback`` and ``callback_sensordata=True``, the same
        refresh is applied to the sensordata passed to the callback after
        each substep.
      control_callback: optional callable
        ``fn(step_index, state, sensordata) -> control`` invoked once before
        every substep, ``nstep`` times total. At ``step_index == 0``,
        ``state`` is the ``initial_state`` array as passed to this method
        and ``sensordata`` is always ``None``: the pool has not re-evaluated
        sensors since the previous call, so callers needing sensordata for
        the first control should reuse the sensordata returned by the
        previous ``step``/``reset``, or call :meth:`forward` first. At
        ``step_index > 0``, ``state`` is the ``(nbatch, nstate)`` float64
        state after the previous substep, and ``sensordata`` is the fresh
        ``(nbatch, nsensordata)`` float64 sensor data when
        ``callback_sensordata`` is true, else ``None``. These arrays are
        internal output buffers that are overwritten between rounds; do not
        mutate them, and copy them if they are needed beyond the callback
        invocation. The returned ``control`` must be a C-contiguous float64
        array of shape ``(nbatch, ncontrol)`` with
        ``ncontrol = mj_stateSize(control_spec)``; it is applied before the
        next substep via ``mj_setState``. The rollout is numerically
        equivalent to calling ``step(nstep=1)`` ``nstep`` times, computing
        control from each returned state. If the callback raises, the
        exception propagates and the rollout is aborted with the pool's
        ``mjData`` workers left in an intermediate state.
      callback_sensordata: when true (default), gather fresh sensordata after
        every substep and pass it to ``control_callback`` for
        ``step_index > 0``. When false, no per-substep sensordata is
        gathered (and no per-substep ``post_step_forward_sensor`` refresh is
        run) and the callback always receives ``None`` for ``sensordata``;
        the final-step sensordata is still gathered and returned when
        ``return_sensor`` is true, with the usual
        ``post_step_forward_sensor`` refresh applied once. Requires
        ``control_callback``; passing ``callback_sensordata=False`` without
        it raises ``ValueError``.

    Returns:
      ``state`` array with shape ``(nbatch, nstate)``, or
      ``(state, sensordata)`` when ``return_sensor`` is true.
    """
    if self._pool is None:
      raise RuntimeError("step requested after pool close")
    if nstep < 1:
      raise ValueError("nstep must be >= 1")
    if control_callback is not None:
      if control is not None:
        raise ValueError("control and control_callback are mutually exclusive")
      if not callable(control_callback):
        raise TypeError("control_callback must be callable")
    elif not callback_sensordata:
      raise ValueError("callback_sensordata=False requires control_callback")

    initial_state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if initial_state.ndim != 2 or initial_state.shape[0] != self.nbatch:
      raise ValueError(
          f"initial_state must have shape (nbatch={self.nbatch}, nstate), "
          f"got {initial_state.shape}"
      )
    if control is not None:
      control = np.ascontiguousarray(control, dtype=np.float64)
    if initial_warmstart is not None:
      initial_warmstart = np.ascontiguousarray(
          initial_warmstart, dtype=np.float64
      )

    return self._pool.step(
        nstep=int(nstep),
        control_spec=int(control_spec),
        state0=initial_state,
        warmstart0=initial_warmstart,
        control=control,
        chunk_size=chunk_size,
        return_sensor=bool(return_sensor),
        post_step_forward_sensor=bool(post_step_forward_sensor),
        control_callback=control_callback,
        callback_sensordata=bool(callback_sensordata),
    )

  # -- forward --------------------------------------------------------
  def forward(
      self,
      initial_state,
      *,
      initial_warmstart=None,
      skipsensor: bool = False,
      chunk_size: Optional[int] = None,
  ):
    """Run a single ``mj_forward`` on every environment.

    Replaces the old ``mujoco.batch_forward`` module.

    Args:
      initial_state: ``(nbatch, nstate)`` full-physics states.
      initial_warmstart: ``(nbatch, nv)`` qacc_warmstart, optional.
      skipsensor: skip sensor evaluation.
      chunk_size: thread-pool chunk size, optional.

    Returns:
      ``sensordata`` array with shape ``(nbatch, nsensordata)``.
    """
    if self._pool is None:
      raise RuntimeError("forward requested after pool close")

    initial_state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if initial_state.ndim != 2 or initial_state.shape[0] != self.nbatch:
      raise ValueError(
          f"initial_state must have shape (nbatch={self.nbatch}, nstate), "
          f"got {initial_state.shape}"
      )
    if initial_warmstart is not None:
      initial_warmstart = np.ascontiguousarray(
          initial_warmstart, dtype=np.float64
      )

    return self._pool.forward(
        state0=initial_state,
        warmstart0=initial_warmstart,
        skipsensor=bool(skipsensor),
        chunk_size=chunk_size,
    )

  # -- batched site Jacobians ----------------------------------------
  def compute_site_jacobians(
      self,
      initial_state,
      site_ids,
      *,
      jacp: bool = True,
      jacr: bool = False,
      initial_warmstart=None,
      chunk_size: Optional[int] = None,
  ):
    """Compute site Jacobians over the full env pool in parallel.

    Per env this runs ``mj_kinematics + mj_comPos`` on the worker's mjData
    (the minimum prefix required by ``mj_jacSite``), then emits jacp / jacr
    for every requested site.

    Args:
      initial_state: ``(nbatch, nstate)`` full-physics states.
      site_ids: scalar int or 1-D int sequence ``(K,)``. Same site set is
        used for every env.
      jacp: emit translation Jacobians ``(nbatch, K, 3, nv)``.
      jacr: emit rotation Jacobians    ``(nbatch, K, 3, nv)``.
      initial_warmstart: ``(nbatch, nv)`` qacc_warmstart, optional.
      chunk_size: thread-pool chunk size, optional.

    Returns:
      A tuple ``(jacp, jacr)``. The unrequested entry is ``None``. When
      ``site_ids`` is a Python scalar the K dimension is squeezed, returning
      ``(nbatch, 3, nv)`` outputs.
    """
    if self._pool is None:
      raise RuntimeError("compute_site_jacobians requested after pool close")
    if not jacp and not jacr:
      raise ValueError("at least one of jacp / jacr must be True")

    initial_state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if initial_state.ndim != 2 or initial_state.shape[0] != self.nbatch:
      raise ValueError(
          f"initial_state must have shape (nbatch={self.nbatch}, nstate), "
          f"got {initial_state.shape}"
      )
    if initial_warmstart is not None:
      initial_warmstart = np.ascontiguousarray(
          initial_warmstart, dtype=np.float64
      )

    site_ids_arr, scalar_index = _normalize_indices(site_ids)

    jacp_arr, jacr_arr = self._pool.compute_site_jacobians(
        state0=initial_state,
        site_ids=site_ids_arr,
        jacp=bool(jacp),
        jacr=bool(jacr),
        warmstart0=initial_warmstart,
        chunk_size=chunk_size,
    )
    if scalar_index:
      if jacp_arr is not None:
        jacp_arr = jacp_arr[:, 0]
      if jacr_arr is not None:
        jacr_arr = jacr_arr[:, 0]
    return jacp_arr, jacr_arr

  # -- hfield height sampling ----------------------------------------
  def sample_hfield_height(
      self,
      initial_state,
      hfield_geom_id: int,
      offsets,
      frame_body_id: int,
      *,
      alignment: str = "yaw",
      output: str = "height",
      chunk_size: Optional[int] = None,
  ) -> np.ndarray:
    """Sample a MuJoCo hfield geom with bilinear interpolation.

    This is a hfield-only sensor primitive. Per environment it sets
    ``initial_state`` on the pool-owned model/data, runs the kinematic prefix
    needed for body and geom poses, attaches ``offsets`` to ``frame_body_id``,
    and samples the target hfield in geom-local coordinates. Samples outside
    the hfield domain are clamped to the border.

    Args:
      initial_state: ``(nbatch, nstate)`` full-physics states.
      hfield_geom_id: integer geom id; the geom must be type ``hfield`` in
        every pool-owned model.
      offsets: ``(npoint, 2)`` local XY sample pattern.
      frame_body_id: body id used as the attachment frame.
      alignment: ``"yaw"`` (default), ``"world"``/``"none"``, or
        ``"body"``/``"full"``.
      output: ``"height"``/``"terrain_height"`` for sampled world z, or
        ``"clearance"`` for ``frame_z - sampled_world_z``.
      chunk_size: thread-pool chunk size, optional.

    Returns:
      ``(nbatch, npoint)`` float64 array.
    """
    if self._pool is None:
      raise RuntimeError("sample_hfield_height requested after pool close")

    initial_state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if initial_state.shape != (self.nbatch, self.nstate):
      raise ValueError(
          f"initial_state must have shape (nbatch={self.nbatch}, "
          f"nstate={self.nstate}), got {initial_state.shape}"
      )

    offsets = np.ascontiguousarray(offsets, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
      raise ValueError(f"offsets must have shape (npoint, 2), got {offsets.shape}")
    if offsets.shape[0] == 0:
      raise ValueError("offsets must be non-empty")

    hfield_geom_id = _normalize_scalar_int("hfield_geom_id", hfield_geom_id)
    frame_body_id = _normalize_scalar_int("frame_body_id", frame_body_id)
    if chunk_size is not None and chunk_size <= 0:
      raise ValueError("chunk_size must be positive")
    alignment = str(alignment).lower()
    output = str(output).lower()

    return self._pool.sample_hfield_height(
        state0=initial_state,
        hfield_geom_id=hfield_geom_id,
        offsets=offsets,
        frame_body_id=frame_body_id,
        alignment=alignment,
        output=output,
        chunk_size=chunk_size,
    )

  # -- sparse reset ---------------------------------------------------
  def reset(
      self,
      env_ids: Sequence[int],
      initial_state,
      *,
      randomization: Optional[Dict[str, Any]] = None,
      initial_warmstart=None,
      skipsensor: bool = False,
      chunk_size: Optional[int] = None,
  ):
    """Reset a subset of environments, optionally applying field patches.

    Args:
      env_ids: 1-D array of environment indices to reset.
      initial_state: ``(len(env_ids), nstate)`` full-physics states.
      randomization: optional ``Dict[str, ndarray]`` mapping field name
        to a payload with leading dim ``len(env_ids)``.
      initial_warmstart: optional ``(len(env_ids), nv)``.
      skipsensor: skip sensor evaluation.
      chunk_size: thread-pool chunk size, optional.

    Returns:
      ``(state, sensordata)`` with leading dim ``len(env_ids)``.
    """
    if self._pool is None:
      raise RuntimeError("reset requested after pool close")

    env_ids_arr = np.ascontiguousarray(env_ids, dtype=np.int32)
    if env_ids_arr.ndim != 1:
      raise ValueError("env_ids must be 1-D")
    n = int(env_ids_arr.shape[0])

    initial_state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if initial_state.shape != (n, self.nstate):
      raise ValueError(
          f"initial_state must have shape ({n}, nstate={self.nstate}), "
          f"got {initial_state.shape}"
      )
    if initial_warmstart is not None:
      initial_warmstart = np.ascontiguousarray(
          initial_warmstart, dtype=np.float64
      )

    if randomization is not None:
      unknown = [k for k in randomization if k not in SUPPORTED_FIELDS]
      if unknown:
        raise ValueError(
            f"Unknown randomization field(s) {unknown}. "
            f"Supported: {SUPPORTED_FIELDS}"
        )
      for key, val in randomization.items():
        arr = np.ascontiguousarray(val, dtype=np.float64)
        if arr.shape[0] != n:
          raise ValueError(
              f"randomization['{key}'] leading dim must be len(env_ids)={n}, "
              f"got {arr.shape[0]}"
          )
        randomization[key] = arr

    return self._pool.reset(
        env_ids=env_ids_arr,
        initial_state=initial_state,
        randomization=randomization,
        initial_warmstart=initial_warmstart,
        skipsensor=bool(skipsensor),
        chunk_size=chunk_size,
    )

  # -- introspection for tests ---------------------------------------
  def get_field(self, env_id: int, name: str) -> np.ndarray:
    """Return a flat copy of the given field for one environment."""
    if self._pool is None:
      raise RuntimeError("get_field requested after pool close")
    return self._pool.get_field(int(env_id), str(name))

  def get_field_indexed(
      self, env_id: int, name: str, indices: int | Sequence[int]
  ):
    """Return selected entries from one field for one environment.

    Scalar fields return a scalar for single-index access and ``(k,)`` for
    multi-index access. Multi-component fields return ``(width,)`` for
    single-index access and ``(k, width)`` for multi-index access.
    """
    if self._pool is None:
      raise RuntimeError("get_field_indexed requested after pool close")
    name = str(name)
    if name not in SUPPORTED_FIELDS:
      raise ValueError(f"Unknown field '{name}'. Supported: {SUPPORTED_FIELDS}")
    indices_arr, scalar_index = _normalize_indices(indices)
    out = self._pool.get_field_indexed(int(env_id), name, indices_arr)
    if not scalar_index:
      return out
    if _FIELD_COMPONENT_WIDTHS[name] == 1:
      return out[0].item()
    return out[0]

  def set_field_indexed(
      self,
      env_id: int,
      name: str,
      indices: int | Sequence[int],
      value,
  ) -> None:
    """Set selected entries from one field for one environment."""
    if self._pool is None:
      raise RuntimeError("set_field_indexed requested after pool close")
    name = str(name)
    if name not in SUPPORTED_FIELDS:
      raise ValueError(f"Unknown field '{name}'. Supported: {SUPPORTED_FIELDS}")
    indices_arr, scalar_index = _normalize_indices(indices)
    value_arr = _normalize_indexed_value(
        name, scalar_index, int(indices_arr.shape[0]), value
    )
    self._pool.set_field_indexed(
        int(env_id), name, indices_arr, value_arr
    )
