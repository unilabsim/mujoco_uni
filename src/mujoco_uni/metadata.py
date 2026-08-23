"""MuJoCoUni package and solver compatibility metadata."""

from __future__ import annotations

__version__ = "0.4.0"

MUJOCO_DEFAULT_VERSION = "3.8.0"
MUJOCO_MIN_VERSION = "3.5.0"
MUJOCO_MAX_VERSION_EXCLUSIVE = "3.11.0"
MUJOCO_VERSION_SPEC = ">=3.5,<3.11"

# Backward-compatible reference version. Runtime compatibility is governed by
# MUJOCO_VERSION_SPEC and the native extension's build-version watchdog.
MUJOCO_VERSION = MUJOCO_DEFAULT_VERSION

__all__ = [
    "MUJOCO_DEFAULT_VERSION",
    "MUJOCO_MAX_VERSION_EXCLUSIVE",
    "MUJOCO_MIN_VERSION",
    "MUJOCO_VERSION",
    "MUJOCO_VERSION_SPEC",
    "__version__",
]
