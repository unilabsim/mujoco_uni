# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MLX simulation-step wrapper backed by native pybind."""

from __future__ import annotations

import atexit
from collections.abc import Sequence
from typing import Optional, Union

import mlx.core as mx
import mujoco
from mujoco import _mlx_step as _native_mlx_step


ModelLike = Union[mujoco.MjModel, Sequence[mujoco.MjModel]]
DataLike = Union[mujoco.MjData, Sequence[mujoco.MjData]]


class MlxStepRunner:
    """Simulation-step runner with an internal parallel worker pool."""

    def __init__(self, *, nthread: Optional[int] = None):
        self.nthread = 0 if nthread is None else int(nthread)
        self._runner = _native_mlx_step.MlxStepRunner(nthread=self.nthread)

    def __enter__(self) -> "MlxStepRunner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._runner is not None:
            del self._runner
            self._runner = None

    def step(
        self,
        model: ModelLike,
        data: DataLike,
        initial_state,
        control=None,
        *,
        control_spec: int = mujoco.mjtState.mjSTATE_CTRL.value,
        nstep: Optional[int] = None,
        initial_warmstart=None,
        chunk_size: Optional[int] = None,
        out_dtype: mx.Dtype = mx.float32,
    ):
        if self._runner is None:
            raise RuntimeError("step requested after thread pool shutdown")

        if not isinstance(model, list):
            model = [model]
        if not isinstance(data, list):
            data = [data]

        if nstep is None:
            if control is None:
                nstep = 1
            else:
                control_arr = mx.array(control)
                nstep = int(control_arr.shape[1]) if control_arr.ndim >= 2 else 1

        return self._runner.step(
            model,
            data,
            int(nstep),
            int(control_spec),
            mx.array(initial_state, dtype=mx.float32),
            None if initial_warmstart is None else mx.array(initial_warmstart, dtype=mx.float32),
            None if control is None else mx.array(control, dtype=mx.float32),
            chunk_size,
            out_dtype,
        )

persistent_mlx_step = None


def shutdown_persistent_pool() -> None:
    global persistent_mlx_step
    if persistent_mlx_step is not None:
        persistent_mlx_step.close()
    persistent_mlx_step = None


atexit.register(shutdown_persistent_pool)


def mlx_step(
    model: ModelLike,
    data: DataLike,
    initial_state,
    control=None,
    *,
    control_spec: int = mujoco.mjtState.mjSTATE_CTRL.value,
    nstep: Optional[int] = None,
    initial_warmstart=None,
    chunk_size: Optional[int] = None,
    out_dtype: mx.Dtype = mx.float32,
):
    if not isinstance(data, list):
        data = [data]
    nthread = len(data) if len(data) > 1 else 0
    with MlxStepRunner(nthread=nthread) as runner:
        return runner.step(
            model=model,
            data=data,
            initial_state=initial_state,
            control=control,
            control_spec=control_spec,
            nstep=nstep,
            initial_warmstart=initial_warmstart,
            chunk_size=chunk_size,
            out_dtype=out_dtype,
        )
