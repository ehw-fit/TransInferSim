VENV_PYTHON := .venv/bin/python
UV := uv
EMC := (echo "ERROR: plug-in build failed."; exit 1)

.PHONY: all install build-accelergy submodules-init clean

all: install

install:
	@echo "--- Preparing virtual environment ---"
	$(UV) sync --all-groups
	$(MAKE) build-accelergy
	@echo "--- Finalizing installation ---"
	$(UV) sync --all-groups

submodules-init:
	@echo "--- Initializing git submodules ---"
	git submodule update --init --recursive

build-accelergy: submodules-init
	@echo "--- Building Accelergy plugins ---"
	cd accelergy_plugins/accelergy-cacti-plug-in && $(MAKE) build > /dev/null 2>&1 || $(EMC)
	cd accelergy_plugins/accelergy-neurosim-plugin && $(UV) run --with setuptools --with wheel python setup.py build_ext > /dev/null 2>&1 || $(EMC)
	@echo "--- All sub-components built successfully ---"

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
