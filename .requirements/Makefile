###########################################################################################
# This makefile is used to manage the python environment for the codebase.
#
# Author: Farouk Boukil
###########################################################################################
PYTHON_CMD:=python

VENV_PATH:=$(PWD)/../.venv
REQ_IN:=requirements.in
REQ_TXT:=requirements.txt

CUDA121_WHEELS:="https://download.pytorch.org/whl/cu121"

#########################
# FILES
#########################

# Create $(REQ_TXT).
# This requires that you have pip-tools installed: python3 -m pip install pip-tools
# Skip this if $(REQ_TXT) already up-to-date.
# Run this if $(REQ_IN) has been updated.
requirements.txt: $(REQ_IN)
	pip-compile --output-file=$(REQ_TXT) --find-links=$(CUDA121_WHEELS) $(REQ_IN)

#########################
# PYTHON ENVIRONMENT
#########################

# Create python environment if it does not exist.
pyenv: $(REQ_TXT)
	rm -rf $(VENV_PATH)
	$(PYTHON_CMD) -m venv $(VENV_PATH)

# Synchronise python environment with $(REQ_TXT)
# Anything in $(REQ_TXT) will be installed in the python environment.
# Anything not in $(REQ_TXT) will be removed from the python environment.
sync_pyenv: $(VENV_PATH) $(REQ_TXT)
	$(VENV_PATH)/bin/pip install --upgrade pip
	$(VENV_PATH)/bin/pip install -r $(REQ_TXT)

# Create the $(REQ_TXT) and the python environment then synchronise the python environment with $(REQ_TXT).
# This is a convenience target that combines the all the steps to create a python environment
# from scratch based on the dependencies in $(REQ_IN) file.
synched_pyenv:
	make pyenv
	make sync_pyenv

synched_pyenv_from_new_reqs:
	make requirements.txt
	make pyenv
	make sync_pyenv