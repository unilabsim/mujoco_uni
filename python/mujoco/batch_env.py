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
    env pool. Returns only the **final** state and sensordata
    (``(nbatch, nstate)`` / ``(nbatch, nsensordata)``), not trajectories.

  * :meth:`BatchEnvPool.forward` — single ``mj_forward`` over all envs.
    Returns only ``sensordata``.

  * :meth:`BatchEnvPool.reset` — fused sparse reset over a subset of
    envs with optional per-env model field patching and selective
    ``mj_setConst`` refresh.

Supported randomization fields are listed in ``SUPPORTED_FIELDS``.
"""

from __future__ import annotations

import numbers
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import mujoco
from mujoco import _batch_env as _native


SUPPORTED_FIELDS = tuple(_native.SUPPORTED_FIELDS)
_FIELD_COMPONENT_WIDTHS = {
    "body_mass": 1,
    "body_ipos": 3,
    "body_iquat": 4,
    "body_inertia": 3,
    "dof_armature": 1,
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
  ):
    """Construct a batch pool from one model or a compatible model sequence.

    Args:
      model: A single ``MjModel`` to clone across the pool, or a compatible
        sequence of ``MjModel`` instances with length ``1`` or ``nbatch``.
      nbatch: Number of environments in the pool.
      nthread: Number of worker threads. ``None`` means ``0``.
    """
    if nbatch <= 0:
      raise ValueError("nbatch must be positive")
    if not isinstance(model, mujoco.MjModel):
      model = list(model)
    self._nthread = 0 if nthread is None else int(nthread)
    self._pool = _native.BatchEnvPool(
        model=model, nbatch=int(nbatch), nthread=self._nthread
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

  def get_all_models(self) -> list[mujoco.MjModel]:
    """Return references to all pool-owned models without copying."""
    if self._pool is None:
      raise RuntimeError("get_all_models requested after pool close")
    return self._pool.get_all_models()

  def get_model(
      self, env_ids: int | Sequence[int]
  ) -> mujoco.MjModel | list[mujoco.MjModel]:
    """Return one or more pool-owned model references without copying."""
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
  ):
    """Run ``nstep`` of ``mj_step`` on every environment.

    Returns only the **final** state after all steps. Sensor computation
    is skipped — use :meth:`forward` afterwards if you need sensors.

    Args:
      initial_state: ``(nbatch, nstate)`` full-physics initial states.
      nstep: number of steps.randomization
      control_spec: MuJoCo ``mjtState`` flags for control.
      control: ``(nbatch, nstep, ncontrol)`` control trajectories, optional.
      initial_warmstart: ``(nbatch, nv)`` qacc_warmstart, optional.
      chunk_size: thread-pool chunk size, optional.

    Returns:
      ``state`` array with shape ``(nbatch, nstate)``.
    """
    if self._pool is None:
      raise RuntimeError("step requested after pool close")
    if nstep < 1:
      raise ValueError("nstep must be >= 1")

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
