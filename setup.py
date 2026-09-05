from __future__ import annotations

from pathlib import Path


def find_mujoco_include_dir(mujoco_dir: Path) -> Path:
    """Return the include root that makes ``<mujoco/mujoco.h>`` resolve.

    Official wheels up to 3.9.x ship headers at ``mujoco/include/mujoco/``;
    since 3.10.0 the Windows wheel nests one level deeper
    (``mujoco/include/mujoco/mujoco/``). Probe the classic layout first so
    behavior on all existing platforms is unchanged, then the nested one.
    """
    candidates = (mujoco_dir / "include", mujoco_dir / "include" / "mujoco")
    for root in candidates:
        if (root / "mujoco" / "mujoco.h").is_file():
            return root
    raise RuntimeError(
        "Could not find mujoco/mujoco.h under any known header layout in "
        f"{mujoco_dir} (tried: {', '.join(str(c) for c in candidates)})"
    )


if __name__ == "__main__":
    import platform
    import re
    import subprocess

    from setuptools import Extension, setup
    from setuptools.command.build_ext import build_ext

    def _make_windows_import_lib(dll: Path, sources: list[str], out_dir: Path) -> Path:
        """Generate a MSVC import library for mujoco.dll.

        Official mujoco wheels ship the DLL and headers but no import library, so
        derive one from the mj*/mju* symbols the native sources reference. Entries
        that end up unreferenced are harmless; a genuinely missing symbol fails at
        link time with an explicit error.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        symbols: set[str] = set()
        for src in sources:
            # utf-8 explicitly: sources contain non-ASCII comments, and Windows
            # would otherwise decode with the locale codepage (cp1252).
            text = Path(src).read_text(encoding="utf-8")
            # Match identifiers, not just call sites: symbols are also used as
            # function pointers, e.g. InterceptMjErrors(mj_copyModel)(...).
            symbols.update(re.findall(r"\b(mju?_[A-Za-z0-9_]+)\b", text))
        def_path = out_dir / "mujoco.def"
        def_path.write_text(
            "LIBRARY mujoco.dll\nEXPORTS\n" + "".join(f"  {s}\n" for s in sorted(symbols)),
            encoding="utf-8",
        )
        implib = out_dir / "mujoco.lib"
        machine = {"AMD64": "x64", "ARM64": "ARM64"}[platform.machine()]
        subprocess.run(
            ["lib", f"/def:{def_path}", f"/out:{implib}", f"/machine:{machine}"], check=True
        )
        return implib

    class BuildExt(build_ext):
        def build_extensions(self) -> None:
            import mujoco
            import numpy
            import pybind11

            self.force = True
            build_mujoco_version = getattr(mujoco, "__version__", "unknown")
            mujoco_dir = Path(mujoco.__file__).resolve().parent
            mujoco_include_dir = find_mujoco_include_dir(mujoco_dir)
            system = platform.system()
            if system == "Windows":
                dll = mujoco_dir / "mujoco.dll"
                if not dll.exists():
                    raise RuntimeError(f"Could not find mujoco.dll in {mujoco_dir}")
            else:
                lib_candidates = sorted(mujoco_dir.glob("libmujoco*"))
                if not lib_candidates:
                    raise RuntimeError(f"Could not find libmujoco in {mujoco_dir}")
                libmujoco = lib_candidates[0]

            for ext in self.extensions:
                ext.include_dirs.extend(
                    [
                        pybind11.get_include(),
                        numpy.get_include(),
                        str(mujoco_include_dir),
                        str(Path(__file__).resolve().parent / "src" / "mujoco_uni" / "native"),
                    ]
                )
                ext.define_macros.append(
                    ("MUJOCO_UNI_BUILD_MUJOCO_VERSION", f'"{build_mujoco_version}"')
                )
                if system == "Windows":
                    ext.extra_objects.append(
                        str(_make_windows_import_lib(dll, ext.sources, Path(self.build_temp)))
                    )
                    ext.extra_compile_args.extend(["/std:c++17"])
                else:
                    ext.extra_objects.append(str(libmujoco))
                if system == "Darwin":
                    ext.extra_compile_args.extend(["-std=c++17", "-stdlib=libc++"])
                    ext.extra_link_args.extend(
                        ["-stdlib=libc++", "-Wl,-rpath,@loader_path/../../mujoco"]
                    )
                elif system == "Linux":
                    ext.extra_compile_args.extend(["-std=c++17"])
                    ext.extra_link_args.extend(["-Wl,-rpath,$ORIGIN/../../mujoco"])
            super().build_extensions()

    setup(
        ext_modules=[
            Extension(
                "mujoco_uni.compiled._batch_env",
                sources=[
                    "src/mujoco_uni/native/batch_env.cc",
                    "src/mujoco_uni/native/threadpool.cc",
                ],
                language="c++",
            )
        ],
        cmdclass={"build_ext": BuildExt},
    )
