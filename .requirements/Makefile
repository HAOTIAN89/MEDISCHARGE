###########################################################################################
# This makefile is used to manage the python environment for the codebase.
#
# Author: Farouk Boukil
###########################################################################################

VENV:=$(PWD)/../.venv

REQ_IN:=requirements.in
REQ_TXT:=requirements.txt
REQ_IN_CPU:=requirements_cpu.in
REQ_TXT_CPU:=requirements_cpu.txt

TORCH_GPU_WHL_LINK:="https://download.pytorch.org/whl/cu118"
TORCH_CPU_WHL_LINK:="https://download.pytorch.org/whl/cpu"

#########################
# FILES
#########################

# Create $(REQ_TXT).
# This requires that you have pip-tools installed: python3 -m pip install pip-tools
# Skip this if $(REQ_TXT) already up-to-date.
# Run this if $(REQ_IN) has been updated.
requirements.txt: $(REQ_IN)
	rm -f $(REQ_TXT) && \
	pip-compile --output-file=$(REQ_TXT) --find-links=$(TORCH_GPU_WHL_LINK) $(REQ_IN)

requirements_cpu.txt: $(REQ_IN_CPU)
	rm -f $(REQ_TXT_CPU) && \
	pip-compile --output-file=$(REQ_TXT_CPU) --find-links=$(TORCH_CPU_WHL_LINK) $(REQ_IN_CPU)


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
sync_pyenv: $(VENV) $(REQ_TXT)
	$(VENV)/bin/pip install --upgrade pip && \
	$(VENV)/bin/pip install -r $(REQ_TXT)

sync_pyenv_cpu: $(VENV) $(REQ_TXT_CPU)
	$(VENV)/bin/pip install --upgrade pip && \
	$(VENV)/bin/pip install -r $(REQ_TXT_CPU)

# Create the $(REQ_TXT) and the python environment then synchronise the python environment with $(REQ_TXT).
# This is a convenience target that combines the all the steps to create a python environment
# from scratch based on the dependencies in $(REQ_IN) file.
synched_pyenv_noreq:
	make requirements.txt && \
	make pyenv && \
	make sync_pyenv

synched_pyenv_cpu_noreq:
	make requirements_cpu.txt && \
	make pyenv && \
	make sync_pyenv_cpu

# Create a python environment and synchronise it with $(REQ_TXT)
# This is a convenience target that combines the all the steps to create a python environment
# except that it assumes that the requirements file has already been created and is up-to-date.
synched_pyenv:
	make pyenv && \
	make sync_pyenv

synched_pyenv_cpu:
	make pyenv && \
	make sync_pyenv_cpu
