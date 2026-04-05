"""Persistent executor for reset-lifecycle domain-randomized rollouts.

This module wraps the pybind-backed ``_domain_randomized_rollout`` pool. The
pool owns:
  * a per-environment ``mjModel`` pool (cloned from a caller-supplied base
    model via ``mj_copyModel``),
  * a per-thread ``mjData`` worker pool,
  * an internal thread pool.

It exposes two execution primitives:

  * :meth:`DomainRandomizedRolloutPool.rollout` — full-batch stepping
    equivalent to :func:`mujoco.rollout.rollout` but running on the owned
    model pool.

  * :meth:`DomainRandomizedRolloutPool.reset_subset` — fused sparse reset:
    apply per-environment ``mjModel`` field patches (e.g. ``body_mass``),
    optionally refresh derived constants via ``mj_setConst``, then run
    ``mj_forward`` and return state + sensor outputs for the subset only.

Supported randomization fields for the first iteration are listed in
``SUPPORTED_FIELDS``. Fields ``body_mass``, ``body_ipos``, ``dof_armature``
trigger ``mj_setConst`` refresh; ``geom_friction``, ``dof_damping`` do not.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import mujoco
from mujoco import _domain_randomized_rollout as _native


SUPPORTED_FIELDS = tuple(_native.SUPPORTED_FIELDS)


class DomainRandomizedRolloutPool:
  """Persistent per-environment model pool with sparse-reset execution."""

  def __init__(
      self,
      model: mujoco.MjModel,
      *,
      nbatch: int,
      nthread: Optional[int] = None,
  ):
    if nbatch <= 0:
      raise ValueError("nbatch must be positive")
    self._nthread = 0 if nthread is None else int(nthread)
    self._pool = _native.DomainRandomizedRolloutPool(
        model=model, nbatch=int(nbatch), nthread=self._nthread
    )

  # Context manager sugar. The C++ destructor cleans up, but explicit
  # close() lets callers release native memory before GC.
  def __enter__(self) -> "DomainRandomizedRolloutPool":
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

  # -- rollout --------------------------------------------------------
  def rollout(
      self,
      initial_state,
      *,
      nstep: int,
      control_spec: int = int(mujoco.mjtState.mjSTATE_CTRL),
      control=None,
      initial_warmstart=None,
      state=None,
      sensordata=None,
      chunk_size: Optional[int] = None,
  ):
    """Run ``nstep`` of ``mj_step`` on every environment.

    Args:
      initial_state: ``(nbatch, nstate)`` full-physics initial states.
      nstep: number of steps.
      control_spec: MuJoCo ``mjtState`` flags specifying what ``control``
        contains.
      control: ``(nbatch, nstep, ncontrol)`` control trajectories, optional.
      initial_warmstart: ``(nbatch, nv)`` qacc_warmstart, optional.
      state: preallocated ``(nbatch, nstep, nstate)`` state output, optional.
      sensordata: preallocated ``(nbatch, nstep, nsensordata)``, optional.
      chunk_size: thread-pool chunk size, optional.

    Returns:
      (state, sensordata) numpy arrays. If ``state``/``sensordata`` were
      passed in they are returned unchanged.
    """
    if self._pool is None:
      raise RuntimeError("rollout requested after pool close")
    if nstep < 1:
      raise ValueError("nstep must be >= 1")

    initial_state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if initial_state.ndim != 2 or initial_state.shape[0] != self.nbatch:
      raise ValueError(
          f"initial_state must have shape (nbatch={self.nbatch}, nstate), "
          f"got {initial_state.shape}"
      )

    nbatch = self.nbatch
    nstate_ = self.nstate
    nv_ = self.nv
    nsensordata_ = self.nsensordata

    if control is not None:
      control = np.ascontiguousarray(control, dtype=np.float64)
    if initial_warmstart is not None:
      initial_warmstart = np.ascontiguousarray(
          initial_warmstart, dtype=np.float64
      )
      if initial_warmstart.shape != (nbatch, nv_):
        raise ValueError(
            f"initial_warmstart must have shape ({nbatch}, {nv_}), "
            f"got {initial_warmstart.shape}"
        )

    if state is None:
      state = np.zeros((nbatch, nstep, nstate_), dtype=np.float64)
    else:
      state = np.ascontiguousarray(state, dtype=np.float64)
    if sensordata is None:
      sensordata = np.zeros((nbatch, nstep, nsensordata_), dtype=np.float64)
    else:
      sensordata = np.ascontiguousarray(sensordata, dtype=np.float64)

    self._pool.rollout(
        nstep=int(nstep),
        control_spec=int(control_spec),
        state0=initial_state,
        warmstart0=initial_warmstart,
        control=control,
        state=state,
        sensordata=sensordata,
        chunk_size=chunk_size,
    )
    return state, sensordata

  # -- sparse reset ---------------------------------------------------
  def reset_subset(
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
      env_ids: iterable of environment indices to reset.
      initial_state: ``(len(env_ids), nstate)`` full-physics states.
      randomization: optional dict mapping field name (one of
        ``SUPPORTED_FIELDS``) to a numpy array with leading dim
        ``len(env_ids)`` and trailing size matching the field for this
        model.
      initial_warmstart: optional ``(len(env_ids), nv)``.
      skipsensor: skip sensor evaluation (see ``mj_forwardSkip``).
      chunk_size: thread-pool chunk size, optional.

    Returns:
      (state, sensordata) with leading dim ``len(env_ids)``. Rows are
      indexed positionally by ``env_ids[k]``; the caller is responsible
      for scattering them into a dense per-env buffer if desired
      (``dense[env_ids] = subset``).
    """
    if self._pool is None:
      raise RuntimeError("reset_subset requested after pool close")

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
      if initial_warmstart.shape != (n, self.nv):
        raise ValueError(
            f"initial_warmstart must have shape ({n}, nv={self.nv}), "
            f"got {initial_warmstart.shape}"
        )

    # Pre-validate randomization shapes in Python — the C++ side will also
    # verify, but we want nicer error messages close to the caller.
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

    return self._pool.reset_subset(
        env_ids=env_ids_arr,
        initial_state=initial_state,
        randomization=randomization,
        initial_warmstart=initial_warmstart,
        skipsensor=bool(skipsensor),
        chunk_size=chunk_size,
    )

  # -- introspection for tests ---------------------------------------
  def get_field(self, env_id: int, name: str) -> np.ndarray:
    """Return a flat copy of the given field for one environment.

    Intended for tests and debugging.
    """
    if self._pool is None:
      raise RuntimeError("get_field requested after pool close")
    return self._pool.get_field(int(env_id), str(name))
