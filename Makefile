# Common development commands for MuJoCoUni (mirrors ../UniLab/Makefile style).

.PHONY: sync
sync:
	uv sync

# Editable install: compiles the native extension against the mujoco version
# currently installed in the active environment.
.PHONY: install
install:
	uv pip install pybind11 wheel setuptools
	uv pip install --force-reinstall --no-deps --no-build-isolation -e .

# Switch the MuJoCo solver version (supported window: >=3.5,<3.12), e.g.
#   make mujoco MJ=3.10.0
# Installs the requested mujoco into the active environment, then rebuilds the
# native extension against it (the extension refuses to load on a version
# mismatch). Afterwards use `make test-no-sync` so uv does not resync the
# environment back to the locked mujoco version.
.PHONY: mujoco
mujoco:
	@test -n "$(MJ)" || (echo "usage: make mujoco MJ=3.10.0" && exit 1)
	uv pip install "mujoco==$(MJ)" pybind11 wheel setuptools
	uv pip install --force-reinstall --no-deps --no-build-isolation -e .

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: test
test:
	uv run pytest -q

# Run tests without an automatic sync, preserving the already-built native
# target after `make mujoco MJ=...`.
.PHONY: test-no-sync
test-no-sync:
	uv run --no-sync pytest -q

.PHONY: check
check: lint test

# Version-matrix checks across MuJoCo 3.5.0 / 3.6.0 / 3.7.0 / 3.8.0 / 3.8.1 / 3.9.0 / 3.10.0 / 3.11.0
.PHONY: matrix
matrix:
	uv run python tools/version_matrix.py --pytest

# Full UniLab task validation (expects a sibling ../UniLab checkout).
.PHONY: test-unilab
test-unilab:
	uv run python tools/version_matrix.py --versions 3.5.0 3.10.0 --unilab

# Short one-iteration UniLab training smoke per MuJoCo environment.
.PHONY: test-unilab-train
test-unilab-train:
	uv run python tools/version_matrix.py --versions 3.5.0 3.10.0 --unilab-train

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist
