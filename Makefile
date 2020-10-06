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
	git subomdule update --init
	ln -s ${PWD}/hopfield-layers/modules ${PWD}/soc/models/hopfield
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
	conda env export -n soc | grep -v "^prefix: " | grep -v libffi | grep -v "en-" > environment_$(OS).yml
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
# Test
###
dump_training_set:
	python scripts/dump.py raw=1 dump_text=1

dump_fixtures:
	python scripts/dump.py --multirun testing=1 raw=1 dump_text=0,1

.PHONY: dump_fixtures dump_training_set


###
# CI
###
typecheck:
	mypy $(CURRENT_DIR)/soc $(CURRENT_DIR)/scripts

lint:
	flake8 soc/. tests/. scripts/.

yapf:
	yapf --style tox.ini -r -i soc/. tests/. scripts/.

TEST_FIXTURE_FILE=$(CURRENT_DIR)/tests/fixtures/soc_seq_3_fullseq.pt
FIXTURES_ZIP_FILE=$(CURRENT_DIR)/tests/fixtures/fixtures_data.zip
FIXTURES_FOLDER=$(CURRENT_DIR)/tests/fixtures
test:
ifeq ("$(wildcard $(TEST_FIXTURE_FILE))","")
	unzip -o $(FIXTURES_ZIP_FILE) -d $(FIXTURES_FOLDER)
	python $(FIXTURES_FOLDER)/dump_preprocessed_files.py
endif
	PYTHONWARNINGS="ignore" pytest .

ci: lint typecheck test

.PHONY: typecheck yapf lint test ci


###
# Experiments
###
exp_clean:
	rm -rf scripts/results/*

exp_hopfield_hp_check:
	cd scripts &&\
	python run_exp.py -m -cn 005_gpu_resnet18fusion_policy_full\
		runner.lr=0.1,0.01,0.001\
		runner.model.n_core_planes=2,8,24\
		runner.model.fusion_num_heads=1,2,8\
		runner.model.beta=0.25,0.5,1.,2.,8.\
		runner.model.update_steps_max=0,3\
		trainer.max_epochs=21

exp_005:
	cd scripts &&\
	python run_exp.py -m -cn 005_gpu_resnet18concat_policy_overfit
	cd scripts &&\
	python run_exp.py -m -cn 005_gpu_resnet18fusion_policy_overfit
	cd scripts &&\
	python run_exp.py -m -cn 005_gpu_resnet18fusion_policy_full
	cd scripts &&\
	python run_exp.py -m -cn 005_gpu_resnet18fusion_policy_pretraining
	cd scripts &&\
	python run_exp.py -m -cn 005_gpu_resnet18fusion_policy_fusionfinetuning


.PHONY: exp_clean


###
# Deploy
###
zip:
	python setup.py sdist --format zip

zip_data:
	zip $(CURRENT_DIR)/data/data.zip $(CURRENT_DIR)/data/*.pt

wheel:
	python setup.py bdist_wheel

clean:
	rm -rf build
	rm -rf dist

.PHONY: zip wheel clean