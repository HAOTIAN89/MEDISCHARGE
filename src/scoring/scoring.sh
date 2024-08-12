#!/bin/bash

# Base directory for input and score directories
base_input_dir="data/formatted"
base_score_dir="data/results"

# Loop through each experiment folder in the input directory
for experiment_dir in "${base_input_dir}"/*; do
  # Extract the experiment name
  experiment_name=$(basename "${experiment_dir}")
  
  # Construct the input and score directories for the current experiment
  input_dir="${base_input_dir}/${experiment_name}"
  score_dir="${base_score_dir}/${experiment_name}"

  # Ensure the score directory exists
  mkdir -p "${score_dir}"

  echo "Scoring experiment: ${experiment_name}"
  # Execute the Python script for the current experiment
  python3 src/scoring/scoring.py --input_dir "${input_dir}" --score_dir "${score_dir}" --metrics bleu rouge meteor bertscore
done