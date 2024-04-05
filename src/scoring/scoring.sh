#!/bin/bash

# Define the base directory for the input and score directories
BASE_INPUT_DIR="data/formatted"
BASE_SCORE_DIR="data/results"

# Iterate over each folder in the base score directory
for folder in ${BASE_SCORE_DIR}/*; do
  if [ -d "${folder}" ]; then
    # Extract just the folder name
    folder_name=$(basename "${folder}")
    
    # Define the input and score directories
    input_dir="${BASE_INPUT_DIR}/${folder_name}"
    score_dir="${BASE_SCORE_DIR}/${folder_name}"
    
    # Check if the input directory exists; if not, skip this folder
    if [ ! -d "${input_dir}" ]; then
      echo "Input directory ${input_dir} does not exist. Skipping."
      continue
    fi

    # Run the Python script with the specified metrics
    echo "Scoring ${folder_name}..."
    python3 src/scoring/scoring.py --input_dir "${input_dir}" --score_dir "${score_dir}" --metrics bleu rouge meteor
  fi
done

echo "Scoring completed for all folders."