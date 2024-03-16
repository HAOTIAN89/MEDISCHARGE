###########################################################################################
# This makefile is used to manage the python environment for the codebase.
#
# Author: Farouk Boukil
###########################################################################################

VENV:=$(PWD)/../.venv

REQ_IN:=requirements.in
REQ_TXT:=requirements.txt

TORCH_LINKS:="https://download.pytorch.org/whl/torch_stable.html"

#########################
# FILES
#########################

# Create $(REQ_TXT).
# This requires that you have pip-tools installed: python3 -m pip install pip-tools
# Skip this if $(REQ_TXT) already up-to-date.
# Run this if $(REQ_IN) has been updated.
requirements.txt: $(REQ_IN)
	rm -f $(REQ_TXT) && \
	pip-compile --output-file=$(REQ_TXT) --find-links=$(TORCH_LINKS) $(REQ_IN)

#########################
# PYTHON ENVIRONMENT
#########################

# Create python environment if it does not exist.
pyenv:
	rm -rf $(VENV) && \
	python3 -m venv $(VENV)

# Synchronise python environment with $(REQ_TXT)
# Anything in $(REQ_TXT) will be installed in the python environment.
# Anything not in $(REQ_TXT) will be removed from the python environment.
sync_pyenv: $(VENV)
	$(VENV)/bin/pip install --upgrade pip && \
	$(VENV)/bin/pip install -r $(REQ_TXT)

# Create the $(REQ_TXT) and the python environment then synchronise the python environment with $(REQ_TXT).
# This is a convenience target that combines the all the steps to create a python environment
# from scratch based on the dependencies in $(REQ_IN) file.
synched_pyenv_noreq: $(VENV)
	make $(REQ_TXT) && \
	make pyenv && \
	make sync_pyenv

# Create a python environment and synchronise it with $(REQ_TXT)
# This is a convenience target that combines the all the steps to create a python environment
# except that it assumes that the requirements file has already been created and is up-to-date.
synched_pyenv: $(VENV)
	make pyenv && \
	make sync_pyenv