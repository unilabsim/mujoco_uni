from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNILAB_ROOT = ROOT.parent / "UniLab"
DEFAULT_VERSIONS = ("3.5.0", "3.6.0", "3.7.0", "3.8.0", "3.8.1", "3.9.0", "3.10.0", "3.11.0")


def _python_path(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _copy_source(dst: Path) -> Path:
    source = dst / "source"

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "dist",
            "__pycache__",
            "mujoco_uni.egg-info",
        }
        for name in names:
            if fnmatch.fnmatch(name, "_batch_env*.so"):
                ignored.add(name)
            if fnmatch.fnmatch(name, "*.pyc"):
                ignored.add(name)
        return ignored.intersection(names)

    shutil.copytree(ROOT, source, ignore=ignore)
    return source


def _smoke_code(expected_version: str) -> str:
    mismatch_version = "3.10.0" if not expected_version.startswith("3.10.") else "3.8.0"
    return f"""
import numpy as np
import mujoco as mj
import subprocess
import sys

import mujoco_uni
from mujoco_uni.batch_env import BatchEnvPool
from mujoco_uni.compiled import MUJOCO_BUILD_VERSION

assert mj.__version__ == {expected_version!r}, mj.__version__
assert MUJOCO_BUILD_VERSION == {expected_version!r}, MUJOCO_BUILD_VERSION
assert mujoco_uni.MUJOCO_VERSION_SPEC == ">=3.5,<3.12"

model = mj.MjModel.from_xml_string(\"\"\"
<mujoco>
  <worldbody>
    <body name="box">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
\"\"\")
data = mj.MjData(model)
state = np.zeros(mj.mj_stateSize(model, int(mj.mjtState.mjSTATE_FULLPHYSICS)))
mj.mj_getState(model, data, state, int(mj.mjtState.mjSTATE_FULLPHYSICS))
states = np.vstack([state, state])

with BatchEnvPool(model, nbatch=2, nthread=1) as pool:
    assert pool.nbatch == 2
    assert pool.nstate == state.shape[0]
    sensor = pool.forward(states)
    stepped = pool.step(states, nstep=1)
    assert sensor.shape == (2, model.nsensordata)
    assert stepped.shape == (2, state.shape[0])

mismatch_code = \"\"\"
import mujoco
mujoco.__version__ = {mismatch_version!r}
try:
    from mujoco_uni.batch_env import BatchEnvPool
except ImportError as exc:
    print(exc)
    raise SystemExit(0)
raise SystemExit("expected build/runtime mismatch watchdog to fail")
\"\"\"
result = subprocess.run(
    [sys.executable, "-c", mismatch_code],
    check=False,
    capture_output=True,
    text=True,
)
assert result.returncode == 0, result.stdout + result.stderr
assert "native batch extension was built against mujoco" in result.stdout

print("ok mujoco", {expected_version!r}, "build", MUJOCO_BUILD_VERSION)
"""


def _run_unilab_checks(version: str, python: Path, source: Path, *, train: bool) -> None:
    if not UNILAB_ROOT.exists():
        raise RuntimeError(f"UniLab checkout not found at {UNILAB_ROOT}")

    del source
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            str(UNILAB_ROOT),
        ]
    )
    _run(
        [
            str(python),
            "-c",
            (
                "import mujoco, mujoco_uni; "
                "from mujoco_uni.compiled import MUJOCO_BUILD_VERSION; "
                f"assert mujoco.__version__ == {version!r}, mujoco.__version__; "
                f"assert MUJOCO_BUILD_VERSION == {version!r}, MUJOCO_BUILD_VERSION; "
                "print('ok unilab env', mujoco.__version__, MUJOCO_BUILD_VERSION)"
            ),
        ]
    )
    # The standalone package tests own the MuJoCoUni version/import contract.
    # The UniLab pass below stays focused on backend behavior and task launch.
    _run(
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests/base/test_mujoco_batch_env_randomization.py",
            "tests/base/test_mujoco_batch_env_jacobian.py",
            "tests/base/backend/test_mujoco_site_jacobian.py",
            "tests/envs/locomotion/test_go2_rough_height_scan.py",
            "tests/envs/locomotion/test_go2_footstand.py",
            "tests/envs/locomotion/test_go2_terrain_spawn.py",
        ],
        cwd=UNILAB_ROOT,
    )
    if train:
        _run(
            [
                str(python),
                "scripts/train_rsl_rl.py",
                "task=go2_joystick_flat/mujoco",
                "algo.seed=1",
                "algo.num_envs=128",
                "algo.num_steps_per_env=24",
                "algo.max_iterations=1",
                "algo.save_interval=100",
                "training.no_play=true",
                "training.logger=tensorboard",
                "training.device=cpu",
                f"training.log_root=logs/mujoco_uni_version_matrix/mj{version}",
            ],
            cwd=UNILAB_ROOT,
        )


def run_version(
    version: str,
    *,
    keep_envs: bool,
    run_pytest: bool,
    run_unilab: bool,
    run_unilab_train: bool,
) -> None:
    tmp = Path(tempfile.mkdtemp(prefix=f"mujoco_uni_mj{version.replace('.', '')}_"))
    try:
        source = _copy_source(tmp)
        env_dir = tmp / ".venv"
        _run(["uv", "venv", str(env_dir), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"])
        python = _python_path(env_dir)
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                f"mujoco=={version}",
                "numpy",
                "pybind11",
                "pytest",
                "wheel",
                "setuptools",
            ]
        )
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--force-reinstall",
                "--no-deps",
                "--no-build-isolation",
                str(source),
            ]
        )
        _run([str(python), "-c", _smoke_code(version)])
        if run_pytest:
            _run(
                [
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_batch_env.py",
                    "tests/test_batch_env_parity.py",
                    "tests/test_version_control.py",
                ],
                cwd=source,
            )
        if run_unilab or run_unilab_train:
            _run_unilab_checks(version, python, source, train=run_unilab_train)
    finally:
        if keep_envs:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MuJoCoUni across MuJoCo versions.")
    parser.add_argument("--versions", nargs="+", default=list(DEFAULT_VERSIONS))
    parser.add_argument("--keep-envs", action="store_true")
    parser.add_argument("--pytest", action="store_true", help="Run the Python test suite per version.")
    parser.add_argument(
        "--unilab",
        action="store_true",
        help="Run focused UniLab MuJoCo integration tests per version.",
    )
    parser.add_argument(
        "--unilab-train",
        action="store_true",
        help="Run a one-iteration UniLab training smoke per version.",
    )
    args = parser.parse_args()

    for version in args.versions:
        print(f"\n== MuJoCo {version} ==", flush=True)
        run_version(
            version,
            keep_envs=bool(args.keep_envs),
            run_pytest=bool(args.pytest),
            run_unilab=bool(args.unilab),
            run_unilab_train=bool(args.unilab_train),
        )


if __name__ == "__main__":
    main()
