SHELL := /usr/bin/env bash

# General dependencies for devbuild, docbuild
REINSTALL_DEPS = $(shell find phidl -type f) venv setup.py

TESTARGS = -s --cov=phidl --cov-config .coveragerc

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

purge: clean
	rm -rf venv

pip-freeze: devbuild
	( \
		source venv/bin/activate; \
		pipdeptree -lf | grep -E '^\w+' | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U; \
		pipdeptree -lf | grep -E '^\w+' | grep -v '^\-e' | grep -v '^#' > dev-requirements.txt; \
	)

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

jupyter: devbuild
	( \
		source venv/bin/activate; \
		cd notebooks; \
		jupyter notebook; \
	)

jupyter-password: venv
	( \
		source venv/bin/activate; \
		jupyter notebook password; \
	)

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "--- environment ---"
	@echo "  venv              creates a python virtualenv in venv/"
	@echo "  pip-freeze        drops all leaf pip packages into dev-requirements.txt (Use with caution)"
	@echo "  clean             clean all build files"
	@echo "  purge             clean and delete virtual environment"
	@echo "--- development ---"
	@echo "  devbuild          install dev dependencies, build phidl, and install inside venv"
	@echo "  jupyter           start a jupyter notebook for development"
	@echo "  jupyter-password  change your jupyter notebook user password"
	@echo "--- testing ---"
	@echo "  testbuild         install test dependencies, build phidl, and install inside venv"
	@echo "  test-unit         perform basic unit tests"
	@echo "--- documentation ---"


.PHONY: help clean purge pip-freeze jupyter test-unit
