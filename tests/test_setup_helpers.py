"""Tests for the setup.py MuJoCo header-layout discovery helper.

Pure-Python: setup.py keeps its setuptools imports under the
``__name__ == "__main__"`` guard so this module imports without setuptools.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_setup_module():
    spec = importlib.util.spec_from_file_location("mujoco_uni_setup", ROOT / "setup.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_mujoco_dir(tmp_path: Path, *headers: str) -> Path:
    mujoco_dir = tmp_path / "mujoco"
    for header in headers:
        header_path = mujoco_dir / header
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text("")
    return mujoco_dir


def test_find_mujoco_include_dir_prefers_classic_layout(tmp_path: Path) -> None:
    setup_module = _load_setup_module()
    mujoco_dir = _make_mujoco_dir(
        tmp_path,
        "include/mujoco/mujoco.h",  # <= 3.9.x, all platforms
        "include/mujoco/mujoco/mujoco.h",  # >= 3.10.0 Windows, must not win
    )
    assert setup_module.find_mujoco_include_dir(mujoco_dir) == mujoco_dir / "include"


def test_find_mujoco_include_dir_accepts_nested_windows_layout(tmp_path: Path) -> None:
    setup_module = _load_setup_module()
    mujoco_dir = _make_mujoco_dir(tmp_path, "include/mujoco/mujoco/mujoco.h")
    assert (
        setup_module.find_mujoco_include_dir(mujoco_dir) == mujoco_dir / "include" / "mujoco"
    )


def test_find_mujoco_include_dir_fails_with_a_clear_error(tmp_path: Path) -> None:
    setup_module = _load_setup_module()
    mujoco_dir = _make_mujoco_dir(tmp_path, "mujoco.dll")
    with pytest.raises(RuntimeError, match="mujoco/mujoco.h"):
        setup_module.find_mujoco_include_dir(mujoco_dir)
