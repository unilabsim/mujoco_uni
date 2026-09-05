"""MuJoCo solver-version control at the runtime boundary.

This module owns the process-level MuJoCo version request, existing versioned
uv environment selection, command spawning, and active official ``mujoco``
runtime validation. It intentionally avoids importing official ``mujoco`` until a
caller passes an already-imported module into ``verify_mujoco_runtime``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, Mapping, Sequence

from mujoco_uni.metadata import (
    MUJOCO_MAX_VERSION_EXCLUSIVE,
    MUJOCO_MIN_VERSION,
    MUJOCO_VERSION_SPEC,
)

SUPPORTED_MUJOCO_MINOR_ORDER = ("3.11", "3.8", "3.10", "3.9", "3.7", "3.6", "3.5")
MUJOCO_UNI_VERSION_ENV = "MUJOCO_UNI_VERSION"


@dataclass(frozen=True)
class MuJoCoEnv:
    """A Python environment with a usable MuJoCo runtime installed."""

    version: str
    env_dir: Path
    python: Path
    source: str = "versioned-env"

    @property
    def minor(self) -> str:
        return minor_version(self.version)


def _is_auto_request(version: str | None) -> bool:
    return version is None or str(version).strip().lower() == "auto"


def requested_mujoco_version(env: Mapping[str, str] | None = None) -> str | None:
    """Return the process-level MuJoCoUni version request, if one is set."""

    selected_env = os.environ if env is None else env
    value = selected_env.get(MUJOCO_UNI_VERSION_ENV)
    if value is None:
        return None
    value = value.strip()
    return value or None


def parse_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:\.|$)", str(version).strip())
    if match is None:
        raise ValueError(f"Unsupported MuJoCo version string: {version!r}")
    patch = 0 if match.group(3) is None else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), patch


def minor_version(version: str) -> str:
    major, minor, _ = parse_version(version)
    return f"{major}.{minor}"


def is_supported_mujoco_version(version: str) -> bool:
    parsed = parse_version(version)
    return parse_version(MUJOCO_MIN_VERSION) <= parsed < parse_version(MUJOCO_MAX_VERSION_EXCLUSIVE)


def _version_request_matches(runtime_version: str, requested: str | None) -> bool:
    if requested is None or _is_auto_request(requested):
        return True
    requested_text = str(requested).strip()
    runtime_parts = parse_version(runtime_version)
    try:
        requested_parts = parse_version(requested_text)
    except ValueError as exc:
        raise ImportError(
            f"{MUJOCO_UNI_VERSION_ENV}={requested_text!r} is not a supported "
            "MuJoCo version request"
        ) from exc
    if re.match(r"^\d+\.\d+\.\d+", requested_text):
        return runtime_parts == requested_parts
    return runtime_parts[:2] == requested_parts[:2]


def verify_mujoco_runtime(
    mujoco_module: ModuleType,
    *,
    requested: str | None = None,
) -> str:
    """Validate an already-imported official MuJoCo runtime."""

    runtime_version = str(getattr(mujoco_module, "__version__", ""))
    if not is_supported_mujoco_version(runtime_version):
        raise ImportError(
            f"mujoco_uni supports official mujoco{MUJOCO_VERSION_SPEC}; "
            f"found mujoco {runtime_version!r}"
        )
    active_request = requested_mujoco_version() if requested is None else requested
    if not _version_request_matches(runtime_version, active_request):
        raise ImportError(
            f"{MUJOCO_UNI_VERSION_ENV}={active_request!r} requested, but the active "
            f"official mujoco runtime is {runtime_version!r}. Launch through the "
            "MuJoCoUni versioned environment selector or choose an existing compatible env."
        )
    if not hasattr(mujoco_module.MjModel, "_address"):
        raise ImportError("mujoco.MjModel._address is required by mujoco_uni")
    if not hasattr(mujoco_module.MjModel, "_from_model_ptr"):
        raise ImportError("mujoco.MjModel._from_model_ptr is required by mujoco_uni")
    return runtime_version


def _python_path(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _run(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=None if cwd is None else str(cwd),
        check=True,
        text=True,
        capture_output=capture,
        env=None if env is None else dict(env),
    )


def _mujoco_version_from_python(python: Path) -> str | None:
    code = (
        "import json, mujoco; "
        "print(json.dumps({'version': getattr(mujoco, '__version__', '')}))"
    )
    try:
        result = _run([python, "-c", code], capture=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        version = str(json.loads(result.stdout)["version"])
    except (KeyError, json.JSONDecodeError, TypeError):
        return None
    return version if is_supported_mujoco_version(version) else None


def _env_from_dir(env_dir: Path, *, source: str = "versioned-env") -> MuJoCoEnv | None:
    python = _python_path(env_dir).absolute()
    if not python.exists():
        return None
    version = _mujoco_version_from_python(python)
    if version is None:
        return None
    return MuJoCoEnv(version=version, env_dir=env_dir, python=python, source=source)


def discover_mujoco_envs(
    roots: Iterable[str | os.PathLike[str]] | None = None,
    *,
    include_current: bool = True,
) -> list[MuJoCoEnv]:
    """Discover versioned uv environments containing supported MuJoCo builds."""

    search_roots = [Path.cwd()] if roots is None else [Path(root) for root in roots]
    envs: list[MuJoCoEnv] = []
    seen_python: set[Path] = set()

    if include_current:
        current_python = Path(sys.executable).absolute()
        version = _mujoco_version_from_python(current_python)
        if version is not None:
            envs.append(
                MuJoCoEnv(
                    version=version,
                    env_dir=current_python.parents[1],
                    python=current_python,
                    source="current-python",
                )
            )
            seen_python.add(current_python)

    for root in search_roots:
        if not root.exists():
            continue
        for env_dir in sorted(root.glob(".venv-mj*")):
            env = _env_from_dir(env_dir)
            if env is None or env.python in seen_python:
                continue
            envs.append(env)
            seen_python.add(env.python)
    return envs


def _minor_preference_index(version: str) -> int:
    minor = minor_version(version)
    try:
        return SUPPORTED_MUJOCO_MINOR_ORDER.index(minor)
    except ValueError:
        return len(SUPPORTED_MUJOCO_MINOR_ORDER)


def _patch_sort_key(version: str) -> int:
    return parse_version(version)[2]


def _ordered_envs(envs: Sequence[MuJoCoEnv]) -> list[MuJoCoEnv]:
    return sorted(
        envs,
        key=lambda env: (_minor_preference_index(env.version), -_patch_sort_key(env.version)),
    )


def _matching_envs(envs: Sequence[MuJoCoEnv], requested: str | None = None) -> list[MuJoCoEnv]:
    if _is_auto_request(requested):
        return _ordered_envs(envs)

    requested_text = str(requested).strip()
    requested_parts = parse_version(requested_text)
    requested_minor = f"{requested_parts[0]}.{requested_parts[1]}"
    exact = bool(re.match(r"^\d+\.\d+\.\d+", requested_text))
    matches = [
        env
        for env in envs
        if (parse_version(env.version) == requested_parts if exact else env.minor == requested_minor)
    ]
    return _ordered_envs(matches)


def _env_for_selected_runtime(env: MuJoCoEnv) -> dict[str, str]:
    selected_env = os.environ.copy()
    selected_env[MUJOCO_UNI_VERSION_ENV] = env.version
    return selected_env


def verify_env(env: MuJoCoEnv) -> None:
    code = f"""
import mujoco
import mujoco_uni
from mujoco_uni.compiled import MUJOCO_BUILD_VERSION
from mujoco_uni.mujoco_runtime import api as mujoco_api
assert mujoco.__version__ == {env.version!r}, mujoco.__version__
assert mujoco_api.__version__ == {env.version!r}, mujoco_api.__version__
assert MUJOCO_BUILD_VERSION == {env.version!r}, MUJOCO_BUILD_VERSION
assert mujoco_uni.MUJOCO_VERSION_SPEC == {MUJOCO_VERSION_SPEC!r}
"""
    _run([env.python, "-c", code], capture=True, env=_env_for_selected_runtime(env))


def _command_for_env(env: MuJoCoEnv, command: Sequence[str]) -> list[str]:
    if not command:
        raise ValueError("command must not be empty")
    first = command[0]
    if Path(first).name.startswith("python"):
        return [str(env.python), *command[1:]]
    return [*map(str, command)]


def run_in_env(
    command: Sequence[str],
    *,
    version: str | None = None,
    env_dir: str | os.PathLike[str] | None = None,
    env_root: str | os.PathLike[str] | None = None,
) -> int:
    """Run a command under an existing MuJoCo versioned environment."""

    if env_dir is not None:
        exact_env = _env_from_dir(Path(env_dir))
        envs = [] if exact_env is None else [exact_env]
    else:
        envs = discover_mujoco_envs(
            [Path.cwd() if env_root is None else Path(env_root)],
            include_current=False,
        )
    if not envs:
        raise RuntimeError(
            "No existing MuJoCo environments were found. Create one before launch, "
            "for example .venv-mj311, .venv-mj38, or .venv-mj310."
        )

    primary = _matching_envs(envs, version)
    requested_missing = False
    if not primary:
        requested_missing = True
        primary = _ordered_envs(envs)

    attempted: set[Path] = set()
    verification_errors: list[str] = []
    selected: MuJoCoEnv | None = None
    for candidate in [*primary, *[env for env in _ordered_envs(envs) if env not in primary]]:
        if candidate.python in attempted:
            continue
        attempted.add(candidate.python)
        try:
            verify_env(candidate)
        except (OSError, subprocess.CalledProcessError, ImportError, AssertionError) as exc:
            verification_errors.append(f"{candidate.python}: {exc}")
            continue
        selected = candidate
        break

    if selected is None:
        detail = "\n".join(verification_errors)
        raise RuntimeError(f"No usable MuJoCoUni environment found.\n{detail}")

    env = selected
    if requested_missing:
        print(
            f"[mujoco_uni] requested mujoco {version!r} was not found in existing "
            f"envs; using mujoco {env.version} at {env.python}",
            flush=True,
        )
    elif primary and env not in primary and not _is_auto_request(version):
        print(
            f"[mujoco_uni] requested mujoco {version!r} was found but unusable; "
            f"using mujoco {env.version} at {env.python}",
            flush=True,
        )
    print(f"[mujoco_uni] using mujoco {env.version}: {env.python}", flush=True)
    return subprocess.run(
        _command_for_env(env, command),
        check=False,
        env=_env_for_selected_runtime(env),
    ).returncode


__all__ = [
    "MUJOCO_UNI_VERSION_ENV",
    "MuJoCoEnv",
    "SUPPORTED_MUJOCO_MINOR_ORDER",
    "discover_mujoco_envs",
    "is_supported_mujoco_version",
    "minor_version",
    "parse_version",
    "requested_mujoco_version",
    "run_in_env",
    "verify_env",
    "verify_mujoco_runtime",
]
