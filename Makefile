SHELL := /usr/bin/env bash
# Makefile has convenience targets for
# 1. Creating virtual environment which has the source version of phidl installed
# 2. Linking phidl user-wide to this source
# 2. Making documentation
# 3. Running tests

# The environment is important if you have a stable version of phidl elsewhere and still want to develop on this source
# Instead of a virtual environment, you can create a user-wide dynamic link to the source by running
# `make dynamic-install`

# DOCTYPE_DEFAULT can be html or latexpdf
DOCTYPE_DEFAULT = html

# General dependencies for devbuild, docbuild
REINSTALL_DEPS = $(shell find phidl -type f) venv setup.py

TESTARGS = -s --cov=phidl --cov-config .coveragerc
TESTARGSNB = --nbval-lax

venv: venv/bin/activate
venv/bin/activate:
	test -d venv || virtualenv -p python3 --prompt "(phidl-venv) " --distribute venv
	touch venv/bin/activate

devbuild: venvinfo/devreqs~
venvinfo/devreqs~: $(REINSTALL_DEPS) dev-requirements.txt
	( \
		source venv/bin/activate; \
		pip install -r dev-requirements.txt | grep -v 'Requirement already satisfied'; \
		pip install -e . | grep -v 'Requirement already satisfied'; \
	)
	@mkdir -p venvinfo
	touch venvinfo/devreqs~

clean:
	rm -rf dist
	rm -rf phidl.egg-info
	rm -rf build
	rm -rf venvinfo
	rm -rf .pytest_cache
	$(MAKE) -C docs clean

purge: clean
	rm -rf venv

pip-freeze: devbuild
	( \
		source venv/bin/activate; \
		pipdeptree -lf | grep -E '^\w+' | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U; \
		pipdeptree -lf | grep -E '^\w+' | grep -v '^\-e' | grep -v '^#' > dev-requirements.txt; \
	)

# Does not go in virtualenv. Changes phidl installation user-wide
dynamic-install:
	pip uninstall phidl
	pip install -e .

testbuild: venvinfo/testreqs~
venvinfo/testreqs~: $(REINSTALL_DEPS) test-requirements.txt
	( \
		source venv/bin/activate; \
		pip install -r test-requirements.txt | grep -v 'Requirement already satisfied'; \
		pip install -e . | grep -v 'Requirement already satisfied'; \
	)
	@mkdir -p venvinfo
	touch venvinfo/testreqs~

test-unit: testbuild
	( \
		source venv/bin/activate; \
		py.test $(TESTARGS) tests; \
	)

docbuild: venvinfo/docreqs~
venvinfo/docreqs~: $(REINSTALL_DEPS) doc-requirements.txt
	( \
		source venv/bin/activate; \
		pip install -r doc-requirements.txt | grep -v 'Requirement already satisfied'; \
		pip install -e . | grep -v 'Requirement already satisfied'; \
	)
	@mkdir -p venvinfo
	@touch venvinfo/docreqs~

docs: docbuild
	source venv/bin/activate; $(MAKE) -C docs $(DOCTYPE_DEFAULT)


help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "--- environment ---"
	@echo "  venv              creates a python virtualenv in venv/"
	@echo "  pip-freeze        drops all leaf pip packages into dev-requirements.txt (Use with caution)"
	@echo "  clean             clean all build files"
	@echo "  purge             clean and delete virtual environment"
	@echo "  dynamic-install   have pip dynamically link phidl to this source code, everywhere on your computer"
	@echo "--- development ---"
	@echo "  devbuild          install dev dependencies, build phidl, and install inside venv"
	@echo "--- testing ---"
	@echo "  testbuild         install test dependencies, build phidl, and install inside venv"
	@echo "  test-unit         perform basic unit tests"
	@echo "--- documentation ---"
	@echo "  docbuild          install doc dependencies, rebuild phidl, and install in venv"
	@echo "  docs              build documentation"


.PHONY: help docs clean purge pip-freeze test-unit
