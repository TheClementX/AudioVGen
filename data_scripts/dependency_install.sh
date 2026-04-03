#!/bin/bash

#create environment
conda env create -f ./environment.yml

#mamba
conda activate AudioVGen
conda install -c nvidia cuda-toolkit=13.0
MAX_JOBS=8 pip causal-conv1d>=1.4.0 --no-build-isolation
MAX_JOBS=8 pip install mamba-ssm --no-build-isolation