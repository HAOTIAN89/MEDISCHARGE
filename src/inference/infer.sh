#!/bin/bash

INFER_DIR=  ...
INPUT_PATH= ...
OUTPUT_PATH= ...
NUM_SAMPLES= ...
VERBOSE= ...

# Models: Meditron 7B, Meditron 13B
if [ "$1" == "meditron-7b" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b \
        --model_path /pure-mlo-scratch/trial-runs/meditron-7b/hf_checkpoint \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/ \
        --num_samples $NUM_SAMPLES \
        --verbose $VERBOSE
fi     
