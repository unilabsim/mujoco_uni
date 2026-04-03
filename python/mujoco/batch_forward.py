"""Native batched mj_forward wrapper backed by C++ pybind."""

from __future__ import annotations

import atexit
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import mujoco
from mujoco import _batch_forward as _native_batch_forward


ModelLike = Union[mujoco.MjModel, Sequence[mujoco.MjModel]]
DataLike = Union[mujoco.MjData, Sequence[mujoco.MjData]]


class BatchForwardRunner:
  """Batch mj_forward runner with an internal parallel worker pool."""

  def __init__(self, *, nthread: Optional[int] = None):
    self.nthread = 0 if nthread is None else int(nthread)
    self._runner = _native_batch_forward.BatchForwardRunner(
        nthread=self.nthread
    )

  def __enter__(self) -> "BatchForwardRunner":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()

  def close(self) -> None:
    if self._runner is not None:
      del self._runner
      self._runner = None

  def forward(
      self,
      model: ModelLike,
      data: DataLike,
      initial_state,
      *,
      initial_warmstart=None,
      chunk_size: Optional[int] = None,
      skipsensor: bool = False,
      out_dtype=np.float64,
      return_state: bool = True,
  ):
    if self._runner is None:
      raise RuntimeError("forward requested after thread pool shutdown")

    if isinstance(model, mujoco.MjModel):
      model = [model]
    elif not isinstance(model, list):
      model = list(model)
    if not isinstance(data, list):
      data = [data]

    state0 = np.asarray(initial_state, dtype=np.float64)
    nbatch = int(state0.shape[0])
    if len(model) == 1 and nbatch > 1:
      model = model * nbatch

    state_np, sensor_np = self._runner.forward(
        model,
        data,
        state0,
        (
            None
            if initial_warmstart is None
            else np.asarray(initial_warmstart, dtype=np.float64)
        ),
        chunk_size,
        bool(skipsensor),
    )

    if out_dtype is not None:
      state_np = np.asarray(state_np, dtype=out_dtype)
      sensor_np = np.asarray(sensor_np, dtype=out_dtype)

    return (state_np, sensor_np) if return_state else sensor_np


persistent_batch_forward = None


def shutdown_persistent_pool() -> None:
  global persistent_batch_forward
  if persistent_batch_forward is not None:
    persistent_batch_forward.close()
  persistent_batch_forward = None


atexit.register(shutdown_persistent_pool)


def batch_forward(
    model: ModelLike,
    data: DataLike,
    initial_state,
    *,
    initial_warmstart=None,
    chunk_size: Optional[int] = None,
    skipsensor: bool = False,
    out_dtype=np.float64,
    return_state: bool = True,
):
  if not isinstance(data, list):
    data = [data]
  nthread = len(data) if len(data) > 1 else 0
  with BatchForwardRunner(nthread=nthread) as runner:
    return runner.forward(
        model=model,
        data=data,
        initial_state=initial_state,
        initial_warmstart=initial_warmstart,
        chunk_size=chunk_size,
        skipsensor=skipsensor,
        out_dtype=out_dtype,
        return_state=return_state,
    )
