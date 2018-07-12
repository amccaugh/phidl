SHELL := /usr/bin/env bash

# DOCTYPE_DEFAULT can be html or latexpdf
DOCTYPE_DEFAULT = html

# General dependencies for devbuild, docbuild
REINSTALL_DEPS = $(shell find phidl -type f) venv setup.py

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
	$(MAKE) -C docs clean

purge: clean
	rm -rf venv

pip-freeze: devbuild
	( \
		source venv/bin/activate; \
		pipdeptree -lf | grep -E '^\w+' | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U; \
		pipdeptree -lf | grep -E '^\w+' | grep -v '^\-e' | grep -v '^#' > dev-requirements.txt; \
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
	@echo "--- development ---"
	@echo "  devbuild          install dev dependencies, build lightlab, and install inside venv"
	@echo "--- testing ---"
	@echo "--- documentation ---"
	@echo "  docs              build documentation"
	@echo "  dochost           build documentation and start local http server"


.PHONY: help docs clean purge dochost pip-freeze jupyter
