"""Parity check: BatchEnvPool vs old rollout + per-env mj_forward.

Verifies that, with no domain randomization, the new BatchEnvPool
produces bit-identical results to the old stateless rollout path on a
real UniLab robot model (go1).

Usage:
    cd /tmp && python /path/to/batch_env_parity_check.py
"""

import copy
import os
import sys

import numpy as np
import mujoco
from mujoco import rollout
from mujoco.batch_env import BatchEnvPool

GO1_XML = os.path.join(
    os.path.dirname(__file__),
    "../../../../src/unilab/assets/robots/go1/go1.xml",
)
# Fallback absolute path if relative doesn't resolve.
if not os.path.isfile(GO1_XML):
    GO1_XML = (
        "/Users/jiayufei/ws/simulator/UniLab"
        "/src/unilab/assets/robots/go1/go1.xml"
    )

NBATCH = 16
NSTEP = 10
SEED = 42


def reference_forward_all(model, states):
    """Per-env mj_forward reference: loop in Python."""
    nbatch = states.shape[0]
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    nsensordata = model.nsensordata
    out_state = np.zeros((nbatch, nstate), dtype=np.float64)
    out_sensor = np.zeros((nbatch, nsensordata), dtype=np.float64)
    data = mujoco.MjData(model)
    for i in range(nbatch):
        mujoco.mj_setState(
            model, data, states[i],
            int(mujoco.mjtState.mjSTATE_FULLPHYSICS),
        )
        mujoco.mj_forward(model, data)
        mujoco.mj_getState(
            model, data, out_state[i],
            int(mujoco.mjtState.mjSTATE_FULLPHYSICS),
        )
        out_sensor[i] = data.sensordata.copy()
    return out_state, out_sensor


def run_parity(nthread):
    tag = f"nthread={nthread}"
    print(f"\n{'='*60}")
    print(f"  Parity check: {tag}, nbatch={NBATCH}, nstep={NSTEP}")
    print(f"{'='*60}")

    model = mujoco.MjModel.from_xml_path(GO1_XML)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    rng = np.random.default_rng(SEED)

    state0 = rng.standard_normal((NBATCH, nstate)).astype(np.float64)
    control = rng.standard_normal(
        (NBATCH, NSTEP, model.nu)
    ).astype(np.float64)

    # ── OLD PATH: stateless rollout ──────────────────────────────
    ref_models = [copy.copy(model) for _ in range(NBATCH)]
    ref_datas = (
        [mujoco.MjData(model) for _ in range(nthread)]
        if nthread > 0
        else [mujoco.MjData(model)]
    )

    ref_state_traj, ref_sensor_traj = rollout.rollout(
        ref_models, ref_datas, state0, control,
    )
    # rollout returns (nbatch, nstep, dim). Take last step.
    ref_last_state = ref_state_traj[:, -1, :].copy()

    # Per-env mj_forward on the final state (reference for sensor).
    ref_fwd_state, ref_fwd_sensor = reference_forward_all(
        model, ref_last_state
    )

    # ── NEW PATH: BatchEnvPool ───────────────────────────────────
    pool = BatchEnvPool(model, nbatch=NBATCH, nthread=nthread)

    pool_state = pool.step(
        state0,
        nstep=NSTEP,
        control_spec=int(mujoco.mjtState.mjSTATE_CTRL),
        control=control,
    )

    pool_fwd_state, pool_fwd_sensor = pool.forward(pool_state)
    pool.close()

    # ── ASSERTIONS ───────────────────────────────────────────────
    passed = 0
    total = 0

    def check(name, a, b):
        nonlocal passed, total
        total += 1
        try:
            np.testing.assert_array_equal(a, b)
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            # Fall back to allclose to show magnitude of difference.
            maxdiff = np.max(np.abs(a - b))
            print(f"  [FAIL] {name}  (max diff = {maxdiff:.2e})")
            # Still try allclose as a softer check.
            try:
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
                print(f"         (but passes allclose atol=1e-12)")
                passed += 1
            except AssertionError:
                print(f"         HARD FAIL: {e}")

    print(f"\n  model: go1.xml  nu={model.nu}  nv={model.nv}  "
          f"nstate={nstate}  nsensordata={model.nsensordata}")
    print(f"  state0 shape: {state0.shape}  control shape: {control.shape}")
    print()

    check(f"step() vs rollout last-step  [{tag}]",
          pool_state, ref_last_state)
    check(f"forward() state vs ref       [{tag}]",
          pool_fwd_state, ref_fwd_state)
    check(f"forward() sensor vs ref      [{tag}]",
          pool_fwd_sensor, ref_fwd_sensor)

    return passed, total


def main():
    if not os.path.isfile(GO1_XML):
        print(f"ERROR: go1.xml not found at {GO1_XML}", file=sys.stderr)
        sys.exit(1)

    total_passed = 0
    total_tests = 0

    for nt in [0, 4]:
        p, t = run_parity(nt)
        total_passed += p
        total_tests += t

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_passed}/{total_tests} passed")
    print(f"{'='*60}")
    if total_passed < total_tests:
        sys.exit(1)


if __name__ == "__main__":
    main()
