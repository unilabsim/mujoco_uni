# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Install script for MuJoCo."""

import fnmatch
import hashlib
import json
import logging
import os
import platform
import random
import re
import shutil
import string
import subprocess
import sys
import sysconfig

import setuptools
from setuptools.command import build_ext
from setuptools.command import install_scripts

MUJOCO_CMAKE = 'MUJOCO_CMAKE'
MUJOCO_CMAKE_ARGS = 'MUJOCO_CMAKE_ARGS'
MUJOCO_FETCHCONTENT_BASE_DIR = 'MUJOCO_FETCHCONTENT_BASE_DIR'
MUJOCO_PATH = 'MUJOCO_PATH'
MUJOCO_PLUGIN_PATH = 'MUJOCO_PLUGIN_PATH'
MUJOCO_PYTHON_EXTENSIONS = 'MUJOCO_PYTHON_EXTENSIONS'

EXT_PREFIX = 'mujoco.'
DEFAULT_EXTENSIONS = (
    'mujoco._callbacks',
    'mujoco._constants',
    'mujoco._enums',
    'mujoco._errors',
    'mujoco._functions',
    'mujoco._render',
    'mujoco._rollout',
    'mujoco._batch_env',
    'mujoco._simulate',
    'mujoco._specs',
    'mujoco._structs',
)


def get_long_description():
  """Creates a long description for the package from bundled markdown files."""
  current_dir = os.path.dirname('__file__')
  with open(os.path.join(current_dir, 'README.md')) as f:
    description = f.read()
  try:
    with open(os.path.join(current_dir, 'LICENSES_THIRD_PARTY.md')) as f:
      description = f'{description}\n{f.read()}'
  except FileNotFoundError:
    pass
  return description


def get_mujoco_lib_pattern():
  if platform.system() == 'Windows':
    return 'mujoco.lib'
  elif platform.system() == 'Darwin':
    return 'libmujoco.*.dylib'
  else:
    return 'libmujoco.so.*'


def get_external_lib_patterns():
  if platform.system() == 'Windows':
    return ['mujoco.dll']
  elif platform.system() == 'Darwin':
    return ['libmujoco.*.dylib']
  else:
    return ['libmujoco.so.*']


def get_plugin_lib_patterns():
  if platform.system() == 'Windows':
    return ['*.dll']
  elif platform.system() == 'Darwin':
    return ['lib*.dylib']
  else:
    return ['lib*']


def start_and_end(iterable):
  it = iter(iterable)
  while True:
    try:
      first = next(it)
      second = next(it)
      yield first, second
    except StopIteration:
      return


def tokenize_quoted_substr(input_string, quote_char, placeholders=None):
  """Replace quoted substrings with random text placeholders with no spaces."""
  # Matches quote characters not proceded with a backslash.
  pattern = re.compile(r'(?<!\\)' + quote_char)
  quote_positions = [m.start() for m in pattern.finditer(input_string)]
  if len(quote_positions) % 2:
    raise ValueError(f'unbalanced quotes {quote_char}...{quote_char}')

  output_string = ''
  placeholders = placeholders if placeholders is not None else dict()
  prev_end = -1
  for start, end in start_and_end(quote_positions):
    output_string += input_string[prev_end + 1 : start]
    while True:
      placeholder = ''.join(random.choices(string.ascii_lowercase, k=5))
      if placeholder not in input_string and placeholder not in output_string:
        break
    output_string += placeholder
    placeholders[placeholder] = input_string[start + 1 : end]
    prev_end = end
  output_string += input_string[prev_end + 1 :]

  return output_string, placeholders


def parse_cmake_args_from_environ(env_var_name=MUJOCO_CMAKE_ARGS):
  """Parses CMake arguments from an environment variable."""
  raw_args = os.environ.get(env_var_name, '').strip()
  unquoted, placeholders = tokenize_quoted_substr(raw_args, '"')
  unquoted, placeholders = tokenize_quoted_substr(unquoted, "'", placeholders)
  parts = re.split(r'\s+', unquoted.strip())
  out = []
  for part in parts:
    for k, v in placeholders.items():
      part = part.replace(k, v)
    part = part.replace('\\"', '"').replace("\\'", "'")
    if part:
      out.append(part)
  return out


def _default_cache_root():
  if platform.system() == 'Windows':
    return os.path.join(
        os.environ.get(
            'LOCALAPPDATA',
            os.path.join(os.path.expanduser('~'), 'AppData', 'Local'),
        ),
        'mujoco-uni',
        'fetchcontent',
    )
  if platform.system() == 'Darwin':
    return os.path.join(
        os.path.expanduser('~/Library/Caches'),
        'mujoco-uni',
        'fetchcontent',
    )
  return os.path.join(
      os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
      'mujoco-uni',
      'fetchcontent',
  )


def _sanitize_cache_token(value):
  token = re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('-')
  return token or 'unknown'


def get_fetchcontent_base_dir(build_cfg):
  """Returns a stable FetchContent cache directory outside build/."""
  override = os.environ.get(MUJOCO_FETCHCONTENT_BASE_DIR, '').strip()
  if override:
    return os.path.abspath(os.path.expanduser(override))

  platform_tag = (
      os.environ.get('_PYTHON_HOST_PLATFORM')
      or sysconfig.get_platform()
      or platform.machine()
  )
  variant = '-'.join([
      _sanitize_cache_token(platform_tag),
      f'py{sys.version_info.major}{sys.version_info.minor}',
      build_cfg.lower(),
  ])
  fingerprint = hashlib.sha256(
      json.dumps(
          {
              'archflags': os.environ.get('ARCHFLAGS', ''),
              'cmake_args': os.environ.get(MUJOCO_CMAKE_ARGS, ''),
              'cmake_generator': os.environ.get('CMAKE_GENERATOR', ''),
              'cc': os.environ.get('CC', ''),
              'cxx': os.environ.get('CXX', ''),
              'deployment_target': os.environ.get(
                  'MACOSX_DEPLOYMENT_TARGET', ''
              ),
          },
          sort_keys=True,
      ).encode('utf-8')
  ).hexdigest()[:12]
  return os.path.join(_default_cache_root(), f'{variant}-{fingerprint}')


class CMakeExtension(setuptools.Extension):
  """A Python extension that has been prebuilt by CMake.

  We do not want distutils to handle the build process for our extensions, so
  so we pass an empty list to the super constructor.
  """

  def __init__(self, name):
    super().__init__(name, sources=[])


class BuildCMakeExtension(build_ext.build_ext):
  """Uses CMake to build extensions."""

  def run(self):
    self._is_apple = platform.system() == 'Darwin'
    (
        self._mujoco_library_path,
        self._mujoco_include_path,
        self._mujoco_plugins_path,
        self._mujoco_framework_path,
    ) = self._find_mujoco()
    self._mujoco_source_root = (
        os.environ[MUJOCO_PATH]
        if MUJOCO_PATH in os.environ
        else os.path.join(os.path.dirname(__file__), 'mujoco')
    )
    self._configure_cmake()
    for ext in self.extensions:
      assert ext.name.startswith(EXT_PREFIX)
      assert '.' not in ext.name[len(EXT_PREFIX) :]
      self.build_extension(ext)
    self._copy_external_libraries()
    self._copy_mujoco_headers()
    self._copy_plugin_libraries()
    if self._is_apple:
      self._copy_mjpython()

  def _find_mujoco(self):
    """Locate MuJoCo library and headers from env vars or bundled sources."""
    pkg_dir = os.path.join(os.path.dirname(__file__), 'mujoco')

    def _search_root(root):
      library_path = None
      include_path = None
      for directory, subdirs, filenames in os.walk(root):
        if self._is_apple and 'mujoco.framework' in subdirs:
          plugin_path = os.path.join(root, 'plugin')
          return (
              os.path.join(directory, 'mujoco.framework/Versions/A'),
              os.path.join(directory, 'mujoco.framework/Headers'),
              plugin_path,
              directory,
          )
        if fnmatch.filter(filenames, get_mujoco_lib_pattern()):
          library_path = directory
        if os.path.exists(os.path.join(directory, 'mujoco', 'mujoco.h')):
          include_path = directory
        if library_path and include_path:
          plugin_path = os.path.join(root, 'plugin')
          return library_path, include_path, plugin_path, None
      return None

    if MUJOCO_PATH in os.environ and MUJOCO_PLUGIN_PATH in os.environ:
      result = _search_root(os.environ[MUJOCO_PATH])
      if result:
        return result
      raise RuntimeError(
          'Cannot find MuJoCo library and/or include paths in MUJOCO_PATH'
      )

    if os.path.isdir(pkg_dir):
      result = _search_root(pkg_dir)
      if result:
        return result

    raise RuntimeError(
        f'{MUJOCO_PATH} environment variable is not set, and no bundled '
        'MuJoCo found in package. Install from a wheel or set MUJOCO_PATH.'
    )

  def _copy_external_libraries(self):
    dst = os.path.dirname(self.get_ext_fullpath(self.extensions[0].name))
    for directory, _, filenames in os.walk(self._mujoco_source_root):
      for pattern in get_external_lib_patterns():
        for filename in fnmatch.filter(filenames, pattern):
          src_path = os.path.join(directory, filename)
          dst_path = os.path.join(dst, filename)
          if os.path.abspath(src_path) == os.path.abspath(dst_path):
            continue
          shutil.copyfile(src_path, dst_path)

  def _copy_plugin_libraries(self):
    dst = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'plugin',
    )
    os.makedirs(dst, exist_ok=True)
    if not os.path.isdir(self._mujoco_plugins_path):
      return
    for directory, _, filenames in os.walk(self._mujoco_plugins_path):
      for pattern in get_plugin_lib_patterns():
        for filename in fnmatch.filter(filenames, pattern):
          src_path = os.path.join(directory, filename)
          dst_path = os.path.join(dst, filename)
          if os.path.abspath(src_path) == os.path.abspath(dst_path):
            continue
          shutil.copyfile(src_path, dst_path)

  def _copy_mujoco_headers(self):
    dst = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'include/mujoco',
    )
    os.makedirs(dst, exist_ok=True)
    for directory, _, filenames in os.walk(self._mujoco_include_path):
      for filename in fnmatch.filter(filenames, '*.h'):
        src_path = os.path.join(directory, filename)
        dst_path = os.path.join(dst, filename)
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
          continue
        shutil.copyfile(src_path, dst_path)

  def _copy_mjpython(self):
    src_dir = os.path.join(os.path.dirname(__file__), 'mujoco/mjpython')
    src_bin = os.path.join(self.build_temp, 'mjpython')
    if not os.path.exists(src_bin):
      return
    dst_contents_dir = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'MuJoCo_(mjpython).app/Contents',
    )
    os.makedirs(dst_contents_dir, exist_ok=True)
    shutil.copyfile(
        os.path.join(src_dir, 'Info.plist'),
        os.path.join(dst_contents_dir, 'Info.plist'),
    )

    dst_bin_dir = os.path.join(dst_contents_dir, 'MacOS')
    os.makedirs(dst_bin_dir, exist_ok=True)
    shutil.copyfile(
        src_bin,
        os.path.join(dst_bin_dir, 'mjpython'),
    )
    os.chmod(os.path.join(dst_bin_dir, 'mjpython'), 0o755)

    dst_resources_dir = os.path.join(dst_contents_dir, 'Resources')
    os.makedirs(dst_resources_dir, exist_ok=True)
    shutil.copyfile(
        os.path.join(src_dir, 'mjpython.icns'),
        os.path.join(dst_resources_dir, 'mjpython.icns'),
    )

  def _configure_cmake(self):
    """Check for CMake."""
    cmake = os.environ.get(MUJOCO_CMAKE, 'cmake')
    build_cfg = 'Debug' if self.debug else 'Release'
    fetchcontent_base_dir = get_fetchcontent_base_dir(build_cfg)
    cmake_module_path = os.path.join(
        os.path.dirname(__file__), 'mujoco', 'cmake'
    )
    cmake_args = [
        f'-DPython3_ROOT_DIR:PATH={sys.prefix}',
        f'-DPython3_EXECUTABLE:STRING={sys.executable}',
        f'-DCMAKE_MODULE_PATH:PATH={cmake_module_path}',
        f'-DCMAKE_BUILD_TYPE:STRING={build_cfg}',
        f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH={self.build_temp}',
        (
            f'-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL={"OFF" if self.debug else "ON"}'
        ),
        f'-DFETCHCONTENT_BASE_DIR:PATH={fetchcontent_base_dir}',
        '-DCMAKE_Fortran_COMPILER:STRING=',
        '-DBUILD_TESTING:BOOL=OFF',
        '-DMUJOCO_BUILD_TESTS:BOOL=OFF',
        '-DMUJOCO_BUILD_TESTS_WASM:BOOL=OFF',
    ]

    if self._mujoco_framework_path is not None:
      cmake_args.extend([
          f'-DMUJOCO_FRAMEWORK_DIR:PATH={self._mujoco_framework_path}',
      ])
    else:
      cmake_args.extend([
          f'-DMUJOCO_LIBRARY_DIR:PATH={self._mujoco_library_path}',
          f'-DMUJOCO_INCLUDE_DIR:PATH={self._mujoco_include_path}',
      ])

    if platform.system() != 'Windows':
      cmake_args.extend([
          f'-DPython3_LIBRARY={sysconfig.get_paths()["stdlib"]}',
          f'-DPython3_INCLUDE_DIR={sysconfig.get_paths()["include"]}',
      ])
    if platform.system() == 'Darwin' and os.environ.get('ARCHFLAGS'):
      osx_archs = []
      if '-arch x86_64' in os.environ['ARCHFLAGS']:
        osx_archs.append('x86_64')
      if '-arch arm64' in os.environ['ARCHFLAGS']:
        osx_archs.append('arm64')
      cmake_args.append(f'-DCMAKE_OSX_ARCHITECTURES={";".join(osx_archs)}')

    cmake_args.extend(parse_cmake_args_from_environ())
    os.makedirs(self.build_temp, exist_ok=True)
    os.makedirs(fetchcontent_base_dir, exist_ok=True)

    if platform.system() == 'Windows':
      cmake_args = [arg.replace('\\', '/') for arg in cmake_args]

    print(f'Using FetchContent cache directory: {fetchcontent_base_dir}')
    print('Configuring CMake with the following arguments:')
    for arg in cmake_args:
      print(f'    {arg}')
    subprocess.check_call(
        [cmake]
        + cmake_args
        + [os.path.join(os.path.dirname(__file__), 'mujoco')],
        cwd=self.build_temp,
    )

    target_names = [ext.name[len(EXT_PREFIX) :] for ext in self.extensions]
    print(f'Building requested CMake targets: {target_names}')
    subprocess.check_call(
        [cmake, '--build', '.', '--config', build_cfg, '--target']
        + target_names
        + [f'-j{os.cpu_count()}'],
        cwd=self.build_temp,
    )

  def build_extension(self, ext):
    dest_path = self.get_ext_fullpath(ext.name)
    build_path = os.path.join(self.build_temp, os.path.basename(dest_path))
    shutil.copyfile(build_path, dest_path)


class InstallScripts(install_scripts.install_scripts):
  """Strips file extension from executable scripts whose names end in `.py`."""

  def run(self):
    super().run()
    oldfiles = self.outfiles
    files = set(oldfiles)
    self.outfiles = []
    for oldfile in oldfiles:
      if oldfile.endswith('.py'):
        newfile = oldfile[:-3]
      else:
        newfile = oldfile

      renamed = False
      if newfile not in files and not os.path.exists(newfile):
        if not self.dry_run:
          os.rename(oldfile, newfile)
        renamed = True

      if renamed:
        logging.info(
            'Renaming %s script to %s',
            os.path.basename(oldfile),
            os.path.basename(newfile),
        )
        self.outfiles.append(newfile)
        files.remove(oldfile)
        files.add(newfile)
      else:
        self.outfiles.append(oldfile)


def get_extensions():
  """Returns the list of CMake extensions to build."""
  ext_names = list(DEFAULT_EXTENSIONS)
  requested = os.environ.get(MUJOCO_PYTHON_EXTENSIONS, '').strip()
  if requested:
    requested_names = []
    for raw_name in requested.split(','):
      name = raw_name.strip()
      if not name:
        continue
      if not name.startswith(EXT_PREFIX):
        name = f'{EXT_PREFIX}{name}'
      requested_names.append(name)

    unknown = sorted(set(requested_names) - set(DEFAULT_EXTENSIONS))
    if unknown:
      raise ValueError(
          f'Unknown extensions in {MUJOCO_PYTHON_EXTENSIONS}: {unknown}. '
          f'Known extensions: {DEFAULT_EXTENSIONS}'
      )

    ext_names = [name for name in DEFAULT_EXTENSIONS if name in set(requested_names)]

  return [CMakeExtension(name) for name in ext_names]


setuptools.setup(
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    cmdclass=dict(
        build_ext=BuildCMakeExtension,
        install_scripts=InstallScripts,
    ),
    ext_modules=get_extensions(),
    scripts=['mujoco/mjpython/mjpython.py']
    if platform.system() == 'Darwin'
    else [],
)
