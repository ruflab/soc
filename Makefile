# Default
all: test
.PHONY: all

OS := $(shell uname | tr '[:upper:]' '[:lower:]')
CURRENT_DIR=$(shell pwd)

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

###
# Package
###
install:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda env create -f environment_$(OS).yml
	@echo ">>> Conda env created."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

## Export conda environment
update_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, exporting conda environment."
	conda env update --name soc --file environment_$(OS).yml
	@echo ">>> Conda env exported."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

## Export conda environment
export_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, exporting conda environment."
	conda env export -n soc | grep -v "^prefix: " > environment_$(OS).yml
ifeq (darwin,$(OS))
	sed -i '' -e 's/numpy-stubs==0.0.1/git+https:\/\/github.com\/numpy\/numpy-stubs.git/g' environment_$(OS).yml
else
	sed -i -e 's/numpy-stubs==0.0.1/git+https:\/\/github.com\/numpy\/numpy-stubs.git/g' environment_$(OS).yml
endif
	@echo ">>> Conda env exported."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

.PHONY: install export_env

###
# CI
###
typecheck:
	mypy $(CURRENT_DIR)/soc $(CURRENT_DIR)/scripts

lint:
	flake8 soc/. tests/. scripts/.

yapf:
	yapf --style tox.ini -r -i soc/. tests/.

test:
	pytest .

ci: lint typecheck test

.PHONY: typecheck yapf lint test ci
